import os
import pytest
import numpy as np
import h5py as h5
import yaml
import re

from sympy import pi, cos, sin, sqrt, exp, ImmutableDenseMatrix as Matrix, Tuple, lambdify
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres as sp_gmres
from scipy.sparse.linalg import minres as sp_minres
from scipy.sparse.linalg import cg as sp_cg
from scipy.sparse.linalg import bicg as sp_bicg
from scipy.sparse.linalg import bicgstab as sp_bicgstab

from sympde.calculus import grad, dot, inner, div, curl, cross
from sympde.calculus import Transpose, laplace
from sympde.topology import NormalVector
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Domain, Square, Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC
from sympde.core     import Constant
from sympde.expr     import TerminalExpr

from psydac.api.essential_bc   import apply_essential_bc
from psydac.fem.basic          import FemField
from psydac.fem.vector         import ProductFemSpace
from psydac.core.bsplines      import make_knots
from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil
from psydac.linalg.stencil     import *
from psydac.linalg.block       import *
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
from psydac.utilities.utils    import refine_array_1d, animate_field, split_space, split_field
from psydac.linalg.iterative_solvers import cg, pcg, bicg, lsmr


from mpi4py import MPI
# comm = MPI.COMM_WORLD

A = Square('A',bounds1=(0.5, 1.), bounds2=(-1., 0.))
V = ScalarFunctionSpace('V0', A, kind=None)
ne = [2 ** 2, 2 ** 2]
degree = [2, 2]
Ah = discretize(A, ncells=ne, comm=None)
Vh = discretize(V, Ah, degree=degree)
uh = FemField(Vh)

import re
def export(self,**fields):

    assert all(field.space is self for field in fields.values())

    V = self.vector_space
    comm = V.cart.comm if V.parallel else None

    # Get space info
    # To do, use geomtery to get the patch number
    patch = 'patch_1'
    #Space information
    ldim = self.ldim

    symbolic_space = self.symbolic_space
    name_space = symbolic_space.name
    pdim = symbolic_space.domain.dim
    kind = symbolic_space.kind
    periodic = self.periodic
    degree = [self.spaces[i].degree for i in range(ldim)]
    basis = [self.spaces[i].basis for i in range(ldim)]
    knots = [self.spaces[i].knots.tolist() for i in range(ldim)]

    new_space = {'name': name_space, 'kind': str(kind), 'dtype': 'float', 'rational': 'false', 'periodic': periodic,
                 'degree': degree, 'basis': basis, 'knots': knots
                 }

    # YAML
    try:
        f = open("spaces.yml",'r')
        current = yaml.safe_load(f)
        f.close()
    except:
        current = None

    if current is not None: #Make sure we aren't overwriting data
        #assert current['ldim'] == ldim
        assert current['ndim'] == pdim
        try:
            patch_index = [current['patches'][i]['name'] for i in range(len(current['patches']))].index(patch,-1)
        except ValueError:
            patch_index = -1
        if patch_index != -1:
            assert all(current['patches'][patch_index]['scalar_spaces'][i]['name'] != name_space
                       for i in range(len(current['patches'][patch_index]['scalar_spaces'])))
            current['patches'][patch_index]['scalar_spaces'].append(new_space)
        else:
            current['patches'].append({'name' : patch, 'scalar_spaces': [new_space]})

    else:
        current = {'ndim' : pdim, 'patches':[{'name' : patch, 'scalar_spaces': [new_space]}]}


    with open('spaces.yml','w') as fout:
        yaml.dump(current,fout,default_flow_style=None, sort_keys=False)



    #HDF5
    # Multi-dimensional index range local to process
    index = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))

    # Create HDF5 file (in parallel mode if MPI communicator size > 1)
    kwargs = {}
    if comm is not None:
        if comm.size > 1:
            kwargs.update(driver='mpio', comm=comm)
    fh5 = h5.File("fields.h5", mode='a', **kwargs)

    regexp = re.compile('snapshot_(?P<id>\d+)')
    try:
        i = max([int(regexp.search(k).group('id')) for k in fh5.keys() if regexp.search(k) is not None]) + 1
    except:
        i = 0
    snapshot = fh5.create_group(f'snapshot_{i:0>4}/{patch}')
    snapshot.attrs.create('t', data=0., dtype=float)
    snapshot.attrs.create('ts', data=0, dtype=int)


    # Add field coefficients as named datasets
    for name_field, field in fields.items():
        dset = snapshot.create_dataset(f'{name_field}', shape=V.npts, dtype=V.dtype)
        dset.attrs.create('space',data=name_space)
        dset[index] = field.coeffs[index]


    # Close HDF5 file
    fh5.close()





export(Vh,u0=uh)