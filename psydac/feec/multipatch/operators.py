# coding: utf-8

# Conga operators on piecewise (broken) de Rham sequences

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import eye as sparse_id

from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping, AffineMapping
from sympde.topology import Boundary, Interface
from sympde.topology import element_of, elements_of
from sympde.calculus import grad, dot, inner, rot, div
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.api.discretization import discretize
from psydac.api.essential_bc import apply_essential_bc_stencil
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix
from psydac.linalg.iterative_solvers import cg, pcg
from psydac.fem.basic   import FemField

from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl, Projector_L2
from psydac.feec.derivatives import Gradient_2D, ScalarCurl_2D

from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator


def get_patch_index_from_face(domain, face):
    domains = domain.interior.args
    if isinstance(face, Interface):
        raise NotImplementedError("This face is an interface, it has several indices -- I am a machine, I cannot choose. Help.")
    elif isinstance(face, Boundary):
        i = domains.index(face.domain)
    else:
        i = domains.index(face)
    return i


#===============================================================================
class ConformingProjection_V0( FemLinearOperator ):
    """
    Conforming projection from global broken space to conforming global space
    Defined by averaging of interface dofs
    """
    # todo (MCP, 16.03.2021):
    #   - extend to several interfaces
    #   - avoid discretizing a bilinear form
    #   - allow case without interfaces (single or multipatch)
    def __init__(self, V0h, domain_h, hom_bc=False):

        FemLinearOperator.__init__(self, fem_domain=V0h)

        V0             = V0h.symbolic_space
        domain         = V0.domain

        u, v = elements_of(V0, names='u, v')
        expr   = u*v  # dot(u,v)

        Interfaces  = domain.interfaces  # note: interfaces does not include the boundary
        expr_I = ( plus(u)-minus(u) )*( plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(Interfaces, expr_I))


        ah = discretize(a, domain_h, [V0h, V0h])

        self._A = ah.assemble()

        spaces = self._A.domain.spaces

        if isinstance(Interfaces, Interface):
            Interfaces = (Interfaces, )

        for b1 in self._A.blocks:
            for A in b1:
                if A is None:continue
                A[:,:,:,:] = 0

        indices = [slice(None,None)]*domain.dim + [0]*domain.dim

        for i in range(len(self._A.blocks)):
            self._A[i,i][tuple(indices)]  = 1

        if Interfaces is not None:

          for I in Interfaces:

            i_minus = get_patch_index_from_face(domain, I.minus)
            i_plus  = get_patch_index_from_face(domain, I.plus )

            sp_minus = spaces[i_minus]
            sp_plus  = spaces[i_plus]

            s_minus = sp_minus.starts[I.axis]
            e_minus = sp_minus.ends[I.axis]

            s_plus = sp_plus.starts[I.axis]
            e_plus = sp_plus.ends[I.axis]

            d_minus = V0h.spaces[i_minus].degree[I.axis]
            d_plus  = V0h.spaces[i_plus].degree[I.axis]

            indices = [slice(None,None)]*domain.dim + [0]*domain.dim

            indices[I.axis] = e_minus
            self._A[i_minus,i_minus][tuple(indices)] = 1/2

            indices[I.axis] = s_plus
            self._A[i_plus,i_plus][tuple(indices)] = 1/2

            indices[I.axis] = d_minus
            indices[domain.dim + I.axis] = -d_plus
            self._A[i_minus,i_plus][tuple(indices)] = 1/2

            indices[I.axis] = s_plus
            indices[domain.dim + I.axis] = d_minus
            self._A[i_plus,i_minus][tuple(indices)] = 1/2

        if hom_bc:
            for bn in domain.boundary:
                i = get_patch_index_from_face(domain, bn)
                for j in range(len(domain)):
                    if self._A[i,j] is None:continue
                    apply_essential_bc_stencil(self._A[i,j], axis=bn.axis, ext=bn.ext, order=0)

        self._matrix = self._A


#===============================================================================
class ConformingProjection_V1( FemLinearOperator ):
    """
    Conforming projection from global broken space to conforming global space

    proj.dot(v) returns the conforming projection of v, computed by solving linear system

    """
    # todo (MCP, 16.03.2021):
    #   - extend to several interfaces
    #   - avoid discretizing a bilinear form
    #   - allow case without interfaces (single or multipatch)
    def __init__(self, V1h, domain_h, hom_bc=False):

        FemLinearOperator.__init__(self, fem_domain=V1h)

        V1             = V1h.symbolic_space
        domain         = V1.domain

        u, v = elements_of(V1, names='u, v')
        expr   = dot(u,v)

        Interfaces      = domain.interfaces  # note: interfaces does not include the boundary
        expr_I = dot( plus(u)-minus(u) , plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(Interfaces, expr_I))

        ah = discretize(a, domain_h, [V1h, V1h])

        self._A = ah.assemble()

        for b1 in self._A.blocks:
            for b2 in b1:
                if b2 is None:continue
                for b3 in b2.blocks:
                    for A in b3:
                        if A is None:continue
                        A[:,:,:,:] = 0

        spaces = self._A.domain.spaces

        if isinstance(Interfaces, Interface):
            Interfaces = (Interfaces, )

        indices = [slice(None,None)]*domain.dim + [0]*domain.dim

        for i in range(len(self._A.blocks)):
            self._A[i,i][0,0][tuple(indices)]  = 1
            self._A[i,i][1,1][tuple(indices)]  = 1

        # empty list if no interfaces ?
        if Interfaces is not None:

            for I in Interfaces:

                i_minus = get_patch_index_from_face(domain, I.minus)
                i_plus  = get_patch_index_from_face(domain, I.plus )

                indices = [slice(None,None)]*domain.dim + [0]*domain.dim

                sp1    = spaces[i_minus]
                sp2    = spaces[i_plus]

                s11 = sp1.spaces[0].starts[I.axis]
                e11 = sp1.spaces[0].ends[I.axis]
                s12 = sp1.spaces[1].starts[I.axis]
                e12 = sp1.spaces[1].ends[I.axis]

                s21 = sp2.spaces[0].starts[I.axis]
                e21 = sp2.spaces[0].ends[I.axis]
                s22 = sp2.spaces[1].starts[I.axis]
                e22 = sp2.spaces[1].ends[I.axis]

                d11     = V1h.spaces[i_minus].spaces[0].degree[I.axis]
                d12     = V1h.spaces[i_minus].spaces[1].degree[I.axis]

                d21     = V1h.spaces[i_plus].spaces[0].degree[I.axis]
                d22     = V1h.spaces[i_plus].spaces[1].degree[I.axis]

                s_minus = [s11, s12]
                e_minus = [e11, e12]

                s_plus = [s21, s22]
                e_plus = [e21, e22]

                d_minus = [d11, d12]
                d_plus  = [d21, d22]

                for k in range(domain.dim):
                    if k == I.axis:continue

                    indices[I.axis] = e_minus[k]
                    self._A[i_minus,i_minus][k,k][tuple(indices)] = 1/2

                    indices[I.axis] = s_plus[k]
                    self._A[i_plus,i_plus][k,k][tuple(indices)] = 1/2

                    indices[I.axis] = d_minus[k]
                    indices[domain.dim + I.axis] = -d_plus[k]
                    self._A[i_minus,i_plus][k,k][tuple(indices)] = 1/2

                    indices[I.axis] = s_plus[k]
                    indices[domain.dim + I.axis] = d_minus[k]

                    self._A[i_plus,i_minus][k,k][tuple(indices)] = 1/2

        if hom_bc:
            for bn in domain.boundary:
                i = get_patch_index_from_face(domain, bn)
                for j in range(len(domain)):
                    if self._A[i,j] is None:continue
                    apply_essential_bc_stencil(self._A[i,j][1-bn.axis,1-bn.axis], axis=bn.axis, ext=bn.ext, order=0)

        self._matrix = self._A
        # exit()

#===============================================================================
class BrokenMass( FemLinearOperator ):
    """
    Broken mass matrix for a scalar space (seen as a LinearOperator... to be improved)
    # TODO: (MCP 10.03.2021) define them as Hodge FemLinearOperators
    # TODO: (MCP 16.03.2021) define also the inverse Hodge

    """
    def __init__( self, Vh, domain_h, is_scalar):

        FemLinearOperator.__init__(self, fem_domain=Vh)

        V = Vh.symbolic_space
        domain = V.domain
        # domain_h = V0h.domain  # would be nice
        u, v = elements_of(V, names='u, v')
        if is_scalar:
            expr   = u*v
        else:
            expr   = dot(u,v)
        a = BilinearForm((u,v), integral(domain, expr))
        ah = discretize(a, domain_h, [Vh, Vh])
        self._matrix = ah.assemble() #.toarray()


#==============================================================================
class BrokenGradient_2D(FemLinearOperator):

    def __init__(self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V0h, fem_codomain=V1h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix for i, D0i in enumerate(D0s)})

    def transpose(self):
        # todo (MCP): define as the dual differential operator
        return BrokenTransposedGradient_2D(self.fem_domain, self.fem_codomain)

#==============================================================================
class BrokenTransposedGradient_2D( FemLinearOperator ):

    def __init__( self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V0h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix.T for i, D0i in enumerate(D0s)})

    def transpose(self):
        # todo (MCP): discard
        return BrokenGradient_2D(self.fem_codomain, self.fem_domain)


#==============================================================================
class BrokenScalarCurl_2D(FemLinearOperator):
    def __init__(self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V2h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D1i._matrix for i, D1i in enumerate(D1s)})

    def transpose(self):
        return BrokenTransposedScalarCurl_2D(V1h=self.fem_domain, V2h=self.fem_codomain)


#==============================================================================
class BrokenTransposedScalarCurl_2D( FemLinearOperator ):

    def __init__( self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V2h, fem_codomain=V1h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D1i._matrix.T for i, D1i in enumerate(D1s)})

    def transpose(self):
        return BrokenScalarCurl_2D(V1h=self.fem_codomain, V2h=self.fem_domain)



#==============================================================================
from sympy import Tuple

# def multipatch_Moments_Hcurl(f, V1h, domain_h):
def ortho_proj_Hcurl(EE, V1h, domain_h, M1):
    """
    return orthogonal projection of E on V1h, given M1 the mass matrix
    """
    assert isinstance(EE, Tuple)
    V1 = V1h.symbolic_space
    v = element_of(V1, name='v')
    l = LinearForm(v, integral(V1.domain, dot(v,EE)))
    lh = discretize(l, domain_h, V1h)
    b = lh.assemble()
    sol_coeffs, info = pcg(M1.mat(), b, pc="jacobi", tol=1e-10)

    return FemField(V1h, coeffs=sol_coeffs)

#==============================================================================
class Multipatch_Projector_H1:
    """
    to apply the H1 projection (2D) on every patch
    """
    def __init__(self, V0h):

        self._P0s = [Projector_H1(V) for V in V0h.spaces]
        self._V0h  = V0h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        u0s = [P(fun) for P, fun, in zip(self._P0s, funs_log)]

        u0_coeffs = BlockVector(self._V0h.vector_space, \
                blocks = [u0j.coeffs for u0j in u0s])

        return FemField(self._V0h, coeffs = u0_coeffs)

#==============================================================================
class Multipatch_Projector_Hcurl:

    """
    to apply the Hcurl projection (2D) on every patch
    """
    def __init__(self, V1h, nquads=None):

        self._P1s = [Projector_Hcurl(V, nquads=nquads) for V in V1h.spaces]
        self._V1h  = V1h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        E1s = [P(fun) for P, fun, in zip(self._P1s, funs_log)]

        E1_coeffs = BlockVector(self._V1h.vector_space, \
                blocks = [E1j.coeffs for E1j in E1s])

        return FemField(self._V1h, coeffs = E1_coeffs)

#==============================================================================
class Multipatch_Projector_L2:

    """
    to apply the L2 projection (2D) on every patch
    """
    def __init__(self, V2h, nquads=None):

        self._P2s = [Projector_L2(V, nquads=nquads) for V in V2h.spaces]
        self._V2h  = V2h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        B2s = [P(fun) for P, fun, in zip(self._P2s, funs_log)]

        B2_coeffs = BlockVector(self._V2h.vector_space, \
                blocks = [B2j.coeffs for B2j in B2s])

        return FemField(self._V2h, coeffs = B2_coeffs)


#==============================================================================
# some plotting utilities

from psydac.feec.pull_push     import push_2d_h1, push_2d_hcurl, push_2d_hdiv, push_2d_l2


# d,M in mappings.items()

def get_grid_vals_scalar(u, etas, mappings, space_kind='h1'):  #_obj):
    # get the physical field values, given the logical field and the logical grid
    n_patches = len(mappings)
    mappings_list = list(mappings.values())
    u_vals = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        u_vals[k] = np.empty_like(eta_1)
        if isinstance(u,FemField):
            uk_field = u.fields[k]   # todo (MCP): try with u[k].fields?
        else:
            # then field is just callable
            uk_field = u[k]
        if space_kind == 'h1':
            # todo (MCP): add 2d_hcurl_vector
            push_field = lambda eta1, eta2: push_2d_h1(uk_field, eta1, eta2)
        else:
            push_field = lambda eta1, eta2: push_2d_l2(uk_field, eta1, eta2, mapping=mappings_list[k])
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                u_vals[k][i, j] = push_field(x1i, x2j)

    u_vals  = np.concatenate(u_vals, axis=1)

    return u_vals


def get_grid_vals_vector(E, etas, mappings, space_kind='hcurl'):
    # get the physical field values, given the logical field and logical grid
    n_patches = len(mappings)
    mappings_list = list(mappings.values())
    E_x_vals = n_patches*[None]
    E_y_vals = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        E_x_vals[k] = np.empty_like(eta_1)
        E_y_vals[k] = np.empty_like(eta_1)
        if isinstance(E,FemField):
            Ek_field_0 = E[k].fields[0]   # or E.fields[k][0] ?
            Ek_field_1 = E[k].fields[1]
        else:
            # then E field is just callable
            Ek_field_0 = E[k][0]
            Ek_field_1 = E[k][1]
        if space_kind == 'hcurl':
            # todo (MCP): specify 2d_hcurl_scalar in push functions
            push_field = lambda eta1, eta2: push_2d_hcurl(Ek_field_0, Ek_field_1, eta1, eta2, mapping=mappings_list[k])
        else:
            push_field = lambda eta1, eta2: push_2d_hdiv(Ek_field_0, Ek_field_1, eta1, eta2, mapping=mappings_list[k])

        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                E_x_vals[k][i, j], E_y_vals[k][i, j] = push_field(x1i, x2j)
    E_x_vals = np.concatenate(E_x_vals, axis=1)
    E_y_vals = np.concatenate(E_y_vals, axis=1)
    return E_x_vals, E_y_vals


from psydac.utilities.utils    import refine_array_1d
from sympy import lambdify


def get_plotting_grid(mappings, N):

    etas     = [[refine_array_1d( bounds, N ) for bounds in zip(D.min_coords, D.max_coords)] for D in mappings]
    mappings_lambda = [lambdify(M.logical_coordinates, M.expressions) for d,M in mappings.items()]

    pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings_lambda, etas)]
    pcoords  = np.concatenate(pcoords, axis=1)

    xx = pcoords[:,:,0]
    yy = pcoords[:,:,1]

    return etas, xx, yy

def get_patch_knots_gridlines(Vh, N, mappings, plotted_patch=-1):
    # get gridlines for one patch grid

    F = [M.get_callable_mapping() for d,M in mappings.items()]

    if plotted_patch in range(len(mappings)):
        grid_x1 = Vh.spaces[plotted_patch].breaks[0]
        grid_x2 = Vh.spaces[plotted_patch].breaks[1]

        x1 = refine_array_1d(grid_x1, N)
        x2 = refine_array_1d(grid_x2, N)

        x1, x2 = np.meshgrid(x1, x2, indexing='ij')
        x, y = F[plotted_patch](x1, x2)

        gridlines_x1 = (x[:, ::N],   y[:, ::N]  )
        gridlines_x2 = (x[::N, :].T, y[::N, :].T)
        # gridlines = (gridlines_x1, gridlines_x2)
    else:
        gridlines_x1 = None
        gridlines_x2 = None

    return gridlines_x1, gridlines_x2

def my_small_plot(
        title, vals, titles,
        xx, yy,
        gridlines_x1=None,
        gridlines_x2=None,
):

    n_plots = len(vals)
    assert n_plots == len(titles)
    #fig = plt.figure(figsize=(17., 4.8))
    fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
    fig.suptitle(title, fontsize=14)

    for i in range(n_plots):
        ax = fig.add_subplot(1, n_plots, i+1)

        if gridlines_x1 is not None:
            ax.plot(*gridlines_x1, color='k')
            ax.plot(*gridlines_x2, color='k')
        cp = ax.contourf(xx, yy, vals[i], 50, cmap='jet', extend='both')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( titles[i] )

    plt.show()

def union(domains, name):
    assert len(domains)>1
    domain = domains[0]
    for p in domains[1:]:
        domain = domain.join(p, name=name)
    return domain

def set_interfaces(domain, interfaces):
    # todo (MCP): add a check that the two faces coincide
    for I in interfaces:
        domain = domain.join(domain, domain.name, bnd_minus=I[0], bnd_plus=I[1])
    return domain

def get_annulus_fourpatches(r_min, r_max):

    dom_log_1 = Square('dom1',bounds1=(r_min, r_max), bounds2=(0, np.pi/2))
    dom_log_2 = Square('dom2',bounds1=(r_min, r_max), bounds2=(np.pi/2, np.pi))
    dom_log_3 = Square('dom3',bounds1=(r_min, r_max), bounds2=(np.pi, np.pi*3/2))
    dom_log_4 = Square('dom4',bounds1=(r_min, r_max), bounds2=(np.pi*3/2, np.pi*2))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_3 = PolarMapping('M3',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_4 = PolarMapping('M4',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    domain_1     = mapping_1(dom_log_1)
    domain_2     = mapping_2(dom_log_2)
    domain_3     = mapping_3(dom_log_3)
    domain_4     = mapping_4(dom_log_4)

    interfaces = [
        [domain_1.get_boundary(axis=1, ext=1), domain_2.get_boundary(axis=1, ext=-1)],
        [domain_2.get_boundary(axis=1, ext=1), domain_3.get_boundary(axis=1, ext=-1)],
        [domain_3.get_boundary(axis=1, ext=1), domain_4.get_boundary(axis=1, ext=-1)],
        [domain_4.get_boundary(axis=1, ext=1), domain_1.get_boundary(axis=1, ext=-1)]
        ]
    domain = union([domain_1, domain_2, domain_3, domain_4], name = 'domain')
    domain = set_interfaces(domain, interfaces)

    mappings  = {
        dom_log_1.interior:mapping_1,
        dom_log_2.interior:mapping_2,
        dom_log_3.interior:mapping_3,
        dom_log_4.interior:mapping_4
    }  # Q (MCP): purpose of a dict ?

    return domain, mappings


def get_2D_rotation_mapping(name='no_name', c1=0, c2=0, alpha=np.pi/2):

    # AffineMapping:
    # _expressions = {'x': 'c1 + a11*x1 + a12*x2 + a13*x3',
    #                 'y': 'c2 + a21*x1 + a22*x2 + a23*x3',
    #                 'z': 'c3 + a31*x1 + a32*x2 + a33*x3'}

    return AffineMapping(
        name, 2, c1=c1, c2=c2,
        a11=np.cos(alpha), a12=-np.sin(alpha),
        a21=np.sin(alpha), a22=np.cos(alpha),
    )

def get_pretzel(h, r_min, r_max, debug_option=1):
    """
    design pretzel-shaped domain with quarter-annuli and quadrangles
    :param h: offset from axes of quarter-annuli
    :param r_min: smaller radius of quarter-annuli
    :param r_max: larger radius of quarter-annuli
    :return: domain, mappings
    """
    assert 0 < r_min
    assert r_min < r_max

    dr = r_max - r_min
    hr = dr/2
    cr = h+(r_max+r_min)/2

    print("building domain: hr = ", hr, ", h = ", h)

    dom_log_1 = Square('dom1',bounds1=(r_min, r_max), bounds2=(0, np.pi/2))
    mapping_1 = PolarMapping('M1',2, c1= h, c2= h, rmin = 0., rmax=1.)
    domain_1  = mapping_1(dom_log_1)

    # shifted left to match dom_log_2
    dom_log_10 = Square('dom10',bounds1=(r_min, r_max), bounds2=(0, np.pi/2))
    mapping_10 = PolarMapping('M10',2, c1= -h, c2= h, rmin = 0., rmax=1.)
    domain_10  = mapping_10(dom_log_10)

    dom_log_2 = Square('dom2',bounds1=(r_min, r_max), bounds2=(np.pi/2, np.pi))
    mapping_2 = PolarMapping('M2',2, c1= -h, c2= h, rmin = 0., rmax=1.)
    domain_2  = mapping_2(dom_log_2)

    dom_log_3 = Square('dom3',bounds1=(r_min, r_max), bounds2=(np.pi, np.pi*3/2))
    mapping_3 = PolarMapping('M3',2, c1= -h, c2= -h, rmin = 0., rmax=1.)
    domain_3  = mapping_3(dom_log_3)

    dom_log_4 = Square('dom4',bounds1=(r_min, r_max), bounds2=(np.pi*3/2, np.pi*2))
    mapping_4 = PolarMapping('M4',2, c1= h, c2= -h, rmin = 0., rmax=1.)
    domain_4  = mapping_4(dom_log_4)

    dom_log_5 = Square('dom5',bounds1=(-hr,hr) , bounds2=(-h/2, h/2))
    mapping_5 = get_2D_rotation_mapping('M5', c1=h/2, c2=cr , alpha=np.pi/2)
    domain_5  = mapping_5(dom_log_5)

    # shifted left to match dom_log_5
    dom_log_50 = Square('dom50',bounds1=(-hr,hr) , bounds2=(-h/2, h/2))
    mapping_50 = get_2D_rotation_mapping('M50', c1=-3*h/2, c2=cr , alpha=np.pi/2)
    domain_50 = mapping_50(dom_log_50)

    dom_log_6 = Square('dom6',bounds1=(-hr,hr) , bounds2=(-h/2, h/2))
    mapping_6 = get_2D_rotation_mapping('M6', c1=-h/2, c2=cr , alpha=np.pi/2)
    domain_6  = mapping_6(dom_log_6)

    dom_log_7 = Square('dom7',bounds1=(-h, h), bounds2=(-r_max, -r_min))
    mapping_7 = IdentityMapping('M7',2)
    domain_7  = mapping_7(dom_log_7)

    dom_log_8 = Square('dom8',bounds1=(r_min, r_max), bounds2=(-h, h))
    mapping_8 = IdentityMapping('M8',2)
    domain_8  = mapping_8(dom_log_8)

    if debug_option == 1:
        domain = union([domain_1, domain_2, domain_3, domain_4,
                        domain_5, domain_6, domain_7, domain_8], name = 'domain')

        interfaces = [
            [domain_1.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=0, ext=+1)],
            [domain_5.get_boundary(axis=0, ext=-1), domain_2.get_boundary(axis=1, ext=-1)],
            [domain_2.get_boundary(axis=1, ext=+1), domain_3.get_boundary(axis=1, ext=-1)],
            [domain_3.get_boundary(axis=1, ext=+1), domain_4.get_boundary(axis=1, ext=-1)],
            [domain_4.get_boundary(axis=1, ext=+1), domain_1.get_boundary(axis=1, ext=-1)]
            ]

        mappings  = {
            dom_log_1.interior:mapping_1,
            dom_log_2.interior:mapping_2,
            dom_log_3.interior:mapping_3,
            dom_log_4.interior:mapping_4,
            dom_log_5.interior:mapping_5,
            dom_log_6.interior:mapping_6,
            dom_log_7.interior:mapping_7,
            dom_log_8.interior:mapping_8
        }  # Q (MCP): purpose of a dict ?

    elif debug_option == 2:
        domain = union([
            domain_5,
            domain_6,
        ], name = 'domain')

        interfaces = [
            [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_5.interior:mapping_5,
            dom_log_6.interior:mapping_6,
        }  # Q (MCP): purpose of a dict ?


    elif debug_option == 3:
        domain = union([
            domain_1,
            domain_5,
            domain_6,
        ], name = 'domain')

        interfaces = [
            [domain_1.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=1, ext=-1)],
            [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_1.interior:mapping_1,
            dom_log_5.interior:mapping_5,
            dom_log_6.interior:mapping_6,
        }  # Q (MCP): purpose of a dict ?

    elif debug_option == 4:
        domain = union([
            domain_1,
            domain_5,
            domain_6,
            domain_2,
        ], name = 'domain')

        interfaces = [
            [domain_1.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=1, ext=-1)],
            [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=-1)],
            [domain_6.get_boundary(axis=1, ext=+1), domain_2.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_1.interior:mapping_1,
            dom_log_5.interior:mapping_5,
            dom_log_6.interior:mapping_6,
            dom_log_2.interior:mapping_2,
        }  # Q (MCP): purpose of a dict ?


    elif debug_option == 10:
        domain = union([
            domain_10,
            domain_2,
        ], name = 'domain')

        interfaces = [
            [domain_10.get_boundary(axis=1, ext=+1), domain_2.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_10.interior:mapping_10,
            dom_log_2.interior:mapping_2,
        }  # Q (MCP): purpose of a dict ?

    elif debug_option == 11:
        domain = union([
            domain_6,
            domain_2,
        ], name = 'domain')

        interfaces = [
            [domain_6.get_boundary(axis=1, ext=+1), domain_2.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_6.interior:mapping_6,
            dom_log_2.interior:mapping_2,
        }  # Q (MCP): purpose of a dict ?

    elif debug_option == 50:
        domain = union([
            domain_6,
            domain_50,
        ], name = 'domain')

        interfaces = [
            [domain_6.get_boundary(axis=1, ext=+1), domain_50.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_6.interior:mapping_6,
            dom_log_50.interior:mapping_50,
        }  # Q (MCP): purpose of a dict ?

    else:
        raise NotImplementedError

    domain = set_interfaces(domain, interfaces)

    print("int: ", domain.interior)
    print("bound: ", domain.boundary)
    print("len(bound): ", len(domain.boundary))
    print("interfaces: ", domain.interfaces)

    return domain, mappings
