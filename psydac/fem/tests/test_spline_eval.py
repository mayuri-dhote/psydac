import numpy as np
import pytest
from mpi4py import MPI

from psydac.fem.basic      import FemField
from psydac.fem.splines    import SplineSpace
from psydac.fem.tensor     import TensorFemSpace
from psydac.linalg.stencil import StencilVector
from psydac.core.bsplines  import find_span, basis_funs, basis_funs_1st_der, basis_funs_all_ders

def reference_eval_field( tfs, field, *eta , weights=None):
    assert isinstance( field, FemField )
    assert field.space is tfs
    assert len( eta ) == tfs.ldim
    if weights:
        assert weights.space == field.coeffs.space

    bases = []
    index = []

    # Necessary if vector coeffs is distributed across processes
    if not field.coeffs.ghost_regions_in_sync:
        field.coeffs.update_ghost_regions()

    for (x, xlim, space) in zip( eta, tfs.eta_lims, tfs.spaces ):

        knots  = space.knots
        degree = space.degree
        span   =  find_span( knots, degree, x )

        #-------------------------------------------------#
        # Fix span for boundaries between subdomains      #
        #-------------------------------------------------#
        # TODO: Use local knot sequence instead of global #
        #       one to get correct span in all situations #
        #-------------------------------------------------#
        if x == xlim[1] and x != knots[-1-degree]:
            span -= 1
        #-------------------------------------------------#
        basis  = basis_funs( knots, degree, x, span )

        # If needed, rescale B-splines to get M-splines
        if space.basis == 'M':
            basis *= space.scaling_array[span-degree : span+1]

        # Determine local span
        wrap_x   = space.periodic and x > xlim[1]
        loc_span = span - space.nbasis if wrap_x else span

        bases.append( basis )
        index.append( slice( loc_span-degree, loc_span+1 ) )

    # Get contiguous copy of the spline coefficients required for evaluation
    index  = tuple( index )
    coeffs = field.coeffs[index].copy()
    if weights:
        coeffs *= weights[index]

    # Evaluation of multi-dimensional spline
    # TODO: optimize

    # Option 1: contract indices one by one and store intermediate results
    #   - Pros: small number of Python iterations = ldim
    #   - Cons: we create ldim-1 temporary objects of decreasing size
    #
    res = coeffs
    for basis in bases[::-1]:
        res = np.dot( res, basis )

#        # Option 2: cycle over each element of 'coeffs' (touched only once)
#        #   - Pros: no temporary objects are created
#        #   - Cons: large number of Python iterations = number of elements in 'coeffs'
#        #
#        res = 0.0
#        for idx,c in np.ndenumerate( coeffs ):
#            ndbasis = np.prod( [b[i] for i,b in zip( idx, bases )] )
#            res    += c * ndbasis

    return res

def reference_eval_field_gradient( tfs, field, *eta ):
    assert isinstance( field, FemField )
    assert field.space is tfs
    assert len( eta ) == tfs.ldim

    bases_0 = []
    bases_1 = []
    index   = []

    for (x, xlim, space) in zip( eta, tfs.eta_lims, tfs.spaces ):
        
        knots   = space.knots
        degree  = space.degree
        span    =  find_span( knots, degree, x )
        #-------------------------------------------------#
        # Fix span for boundaries between subdomains      #
        #-------------------------------------------------#
        # TODO: Use local knot sequence instead of global #
        #       one to get correct span in all situations #
        #-------------------------------------------------#
        if x == xlim[1] and x != knots[-1-degree]:
            span -= 1
        #-------------------------------------------------#
        basis_0 = basis_funs( knots, degree, x, span )
        basis_1 = basis_funs_1st_der( knots, degree, x, span )

        # If needed, rescale B-splines to get M-splines
        if space.basis == 'M':
            scaling  = space.scaling_array[span-degree : span+1]
            basis_0 *= scaling
            basis_1 *= scaling

        # Determine local span
        wrap_x   = space.periodic and x > xlim[1]
        loc_span = span - space.nbasis if wrap_x else span

        bases_0.append( basis_0 )
        bases_1.append( basis_1 )
        index.append( slice( loc_span-degree, loc_span+1 ) )

    # Get contiguous copy of the spline coefficients required for evaluation
    index  = tuple( index )
    coeffs = field.coeffs[index].copy()

    # Evaluate each component of the gradient using algorithm described in "Option 1" above
    grad = []
    for d in range( tfs.ldim ):
        bases = [(bases_1[d] if i==d else bases_0[i]) for i in range( tfs.ldim )]
        res   = coeffs
        for basis in bases[::-1]:
            res = np.dot( res, basis )
        grad.append( res )

    return grad

def run_spline_eval(comm, domain, ncells, degree, periodic, seed):
    # determinize tests
    np.random.seed(seed)

    dim = len(ncells)
    vdom = [np.array(d) for d in domain]

    # scale at least linearly with the dimension
    random_gridpoints = dim*10
    random_points = dim*100
    random_overall = dim*100

    breaks = [np.linspace(*lims, num=n+1) for lims, n in zip(domain, ncells)]

    Ns = [SplineSpace(degree=d, grid=g, periodic=p, basis='B') \
                                  for d, g, p in zip(degree, breaks, periodic)]
    
    # spline space
    Vp = TensorFemSpace(*Ns, comm=comm)
    Vs = TensorFemSpace(*Ns, comm=comm)

    # build data slices in serial and parallel
    slicep = tuple(slice(pad, -pad) for pad in Vp.vector_space.pads)
    slices = tuple(slice(pad, -pad) for pad in Vs.vector_space.pads)

    # look for the chunk which is on the local parallel process
    subslice = tuple(slice(s,e+1) for s,e in zip(Vp.vector_space.starts, Vp.vector_space.ends))

    coeffs = np.random.random(Vs.vector_space.npts)
    weights = np.random.random(Vs.vector_space.npts)

    coeffss = StencilVector(Vs.vector_space)
    coeffss._data[slices] = coeffs
    coeffsp = StencilVector(Vp.vector_space)
    coeffsp._data[slicep] = coeffs[subslice]

    weightss = StencilVector(Vs.vector_space)
    weightss._data[slices] = weights
    weightsp = StencilVector(Vp.vector_space)
    weightsp._data[slicep] = weights[subslice]

    fieldp = FemField(Vp, coeffs=coeffsp)
    fields = FemField(Vs, coeffs=coeffss)

    breaksize = [len(b) for b in breaks]

    testpoints = []

    # domain corners
    testpoints += vdom

    # grid points
    testpoints += [np.array([g[r] for g, r in zip(breaks, np.random.randint(breaksize))]) for _ in range(random_gridpoints)]
    
    # random points inside the domain
    testpoints += np.random.random((random_points, dim)) * (vdom[1] - vdom[0]) + vdom[0]

    # random points everywhere
    testpoints += np.random.randn((random_overall, dim)) * 3 * (vdom[1] - vdom[0]) + vdom[0]

    for point in testpoints:
        assert np.allclose(Vp.eval_field(fieldp, *point), reference_eval_field(Vs, fields, *point), 1e-12, 1e-12)
        assert np.allclose(Vp.eval_field_gradient(fieldp, *point), reference_eval_field_gradient(Vs, fields, *point), 1e-12, 1e-12)
        assert np.allclose(Vp.eval_field(fieldp, *point, weightsp), reference_eval_field(Vs, fields, *point, weightss), 1e-12, 1e-12)

@pytest.mark.parametrize('domain', [(0, 1), (-2, 3)])
@pytest.mark.parametrize('ncells', [11, 37])
@pytest.mark.parametrize('degree', [2, 3, 4, 5])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('seed', [1,3])
def test_spline_eval_1d_ser(domain, ncells, degree, periodic, seed):
    run_spline_eval(None, [domain], [ncells], [degree], [periodic], seed)

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])              
@pytest.mark.parametrize('ncells', [(10, 9), (27, 15)])              
@pytest.mark.parametrize('degree', [(3, 2), (4, 5)])                 
@pytest.mark.parametrize('periodic', [(True, False), (False, True)])
@pytest.mark.parametrize('seed', [1,3])
def test_spline_eval_2d_ser(domain, ncells, degree, periodic, seed):
    run_spline_eval(None, domain, ncells, degree, periodic, seed) 

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])  
@pytest.mark.parametrize('ncells', [(4, 5, 7)])                       
@pytest.mark.parametrize('degree', [(3, 2, 5), (2, 4, 7)])            
@pytest.mark.parametrize('periodic', [( True, False, False),          
                                      (False,  True, False),
                                      (False, False,  True)])
@pytest.mark.parametrize('seed', [1,3])
def test_spline_eval_3d_ser(domain, ncells, degree, periodic, seed):
    run_spline_eval(None, domain, ncells, degree, periodic, seed) 

@pytest.mark.parametrize('domain', [(0, 1), (-2, 3)])
@pytest.mark.parametrize('ncells', [29, 37])
@pytest.mark.parametrize('degree', [2, 3, 4, 5])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parallel
def test_spline_eval_1d_par(domain, ncells, degree, periodic, seed):
    run_spline_eval(MPI.COMM_WORLD, [domain], [ncells], [degree], [periodic], seed)

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8])])              
@pytest.mark.parametrize('ncells', [(10, 11), (27, 15)])              
@pytest.mark.parametrize('degree', [(3, 2), (4, 5)])                 
@pytest.mark.parametrize('periodic', [(True, False), (False, True)])
@pytest.mark.parametrize('seed', [1,3])
@pytest.mark.parallel
def test_spline_eval_2d_par(domain, ncells, degree, periodic, seed):
    run_spline_eval(MPI.COMM_WORLD, domain, ncells, degree, periodic, seed) 

@pytest.mark.parametrize('domain', [([-2, 3], [6, 8], [-0.5, 0.5])])  
@pytest.mark.parametrize('ncells', [(5, 5, 7)])                       
@pytest.mark.parametrize('degree', [(2, 2, 3)])            
@pytest.mark.parametrize('periodic', [( True, False, False),          
                                      (False,  True, False),
                                      (False, False,  True)])
@pytest.mark.parametrize('seed', [3])
@pytest.mark.parallel
def test_spline_eval_3d_par(domain, ncells, degree, periodic, seed):
    run_spline_eval(MPI.COMM_WORLD, domain, ncells, degree, periodic, seed) 

