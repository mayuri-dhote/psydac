# coding: utf-8

import sys
import os
import importlib
import inspect

import numpy as np

from functools import lru_cache
from mpi4py    import MPI

from sympy import Mul, Tuple, Symbol
from sympy import Mod as sy_Mod, Abs, Range, Symbol, Max
from sympy import Function, Integer

from psydac.pyccel.ast.core import Variable, IndexedVariable
from psydac.pyccel.ast.core import For, Comment
from psydac.pyccel.ast.core import Slice, String
from psydac.pyccel.ast.core import ValuedArgument
from psydac.pyccel.ast.core import Assign
from psydac.pyccel.ast.core import AugAssign
from psydac.pyccel.ast.core import Product
from psydac.pyccel.ast.core import FunctionDef
from psydac.pyccel.ast.core import Import

from psydac.pyccel.ast.datatypes import NativeInteger

from psydac.api.ast.nodes     import FloorDiv
from psydac.api.ast.utilities import variables, math_atoms_as_str
from psydac.api.ast.utilities import build_pyccel_types_decorator

from psydac.fem.splines       import SplineSpace
from psydac.fem.tensor        import TensorFemSpace
from psydac.fem.vector        import ProductFemSpace
from psydac.api.ast.basic     import SplBasic
from psydac.api.printing      import pycode
from psydac.api.settings      import PSYDAC_BACKENDS, PSYDAC_DEFAULT_FOLDER
from psydac.api.utilities     import mkdir_p, touch_init_file, random_string, write_code

from psydac.api.ast.linalg_kernels import transpose_1d, interface_transpose_1d
from psydac.api.ast.linalg_kernels import transpose_2d, interface_transpose_2d
from psydac.api.ast.linalg_kernels import transpose_3d, interface_transpose_3d

#==============================================================================
def variable_to_sympy(x):
    if isinstance(x, Variable) and isinstance(x.dtype, NativeInteger):
        x = Symbol(x.name, integer=True)
    return x

#==============================================================================
def compute_diag_len(p, md, mc, return_padding=False):
    p, md, mc = np.int64(p), np.int64(md), np.int64(mc)
    n = ((np.ceil((p+1)/mc)-1)*md).astype('int')
    ep = np.minimum(0, n-p)
    n = n-ep + p+1
    if return_padding:
        return n.astype('int'), (-ep).astype('int')
    else:
        return n.astype('int')

def Mod(a,b):
    if b == 1:
        return Integer(0)
    else:
        return sy_Mod(a,b)

def toInteger(a):
    if isinstance(a, np.int64):
        return Integer(int(a))
    return a
#==============================================================================

class LinearOperatorDot(SplBasic):

    def __new__(cls, ndim, comm=None, **kwargs):
        if comm is not None:
            assert isinstance(comm, MPI.Comm)
            comm_id = comm.py2f()
        else:
            comm_id = None
        return cls.__hashable_new__(ndim, comm_id, **kwargs)

    @classmethod
    @lru_cache(maxsize=32)
    def __hashable_new__(cls, ndim, comm_id=None, **kwargs):

        # If integer communicator is provided, convert it to mpi4py object
        comm = None if comm_id is None else MPI.COMM_WORLD.f2py(comm_id)

        # Generate random tag, unique for all processes in MPI communicator
        tag = random_string(8)
        if comm is not None:
            tag = comm.bcast(tag, root=0)

        # Create new instance of this class
        obj = SplBasic.__new__(cls, tag, prefix='lo_dot', comm=comm)

        # Initialize instance (code generation happens here)
        backend        = dict(kwargs.pop('backend'))
        code           = obj._initialize(ndim, backend=backend, **kwargs)
        obj._arguments = dict((str(a.name), a) for a in code.arguments)
        obj._code      = code
        obj._folder    = obj._initialize_folder()
        obj._generate_code(backend=backend)
        obj._compile(backend=backend)

        # Return instance
        return obj

    @property
    def func(self):
        return self._func

    @property
    def arguments(self):
        return self._arguments

    @property
    def code(self):
        return self._code

    @property
    def folder(self):
        return self._folder

    def _initialize(self, ndim, **kwargs):

        nrows           = tuple(map(toInteger,kwargs.pop('nrows', variables('n1:%s'%(ndim+1),  'int'))))
        nrows_extra     = tuple(map(toInteger,kwargs.pop('nrows_extra', variables('ne1:%s'%(ndim+1),  'int'))))
        starts          = tuple(map(toInteger,kwargs.pop('starts', variables('s1:%s'%(ndim+1),  'int'))))
        indices1        = variables('i1:%s'%(ndim+1),  'int')
        bb              = variables('b1:%s'%(ndim+1),  'int')
        indices2        = variables('k1:%s'%(ndim+1),  'int')

        v               = variables('v','real')
        x, out          = variables('x, out','real', cls=IndexedVariable, rank=ndim)
        mat             = variables('mat','real', cls=IndexedVariable, rank=2*ndim)

        xshape          = variables('xn1:%s'%(ndim+1),  'int')
        backend         = kwargs.pop('backend', None)

        pads            = tuple(map(toInteger, kwargs.pop('pads')))
        gpads           = tuple(map(toInteger,kwargs.pop('gpads')))
        cm              = tuple(map(toInteger,kwargs.pop('cm')))
        dm              = tuple(map(toInteger,kwargs.pop('dm')))
        interface       = kwargs.pop('interface', False)
        flip_axis       = kwargs.pop('flip_axis',[1]*ndim)
        interface_axis  = kwargs.pop('interface_axis', None)
        d_start         = toInteger(kwargs.pop('d_start', None))
        c_start         = toInteger(kwargs.pop('c_start', None))

        openmp          = False if backend is None else backend["openmp"]

        ndiags, _ = list(zip(*[compute_diag_len(p,mj,mi, return_padding=True) for p,mi,mj in zip(pads,cm,dm)]))

        inits  = [Assign(b,p*m+p+1-n-Mod(s,m)) for b,p,m,n,s in zip(bb, gpads, dm, ndiags, starts) if not isinstance(p*m+p+1-n-Mod(s,m),(int,np.int64, Integer))]

        if any(f==-1 for f in flip_axis):
            inits.append(Assign(xshape, Function('shape')(x)))

        bb     = [b if not isinstance(p*m+p+1-n-Mod(s,m),(int,np.int64, Integer)) else p*m+p+1-n-Mod(s,m) for b,p,m,n,s in zip(bb, gpads, dm, ndiags, starts)]
        body   = []

        ranges = [Range(variable_to_sympy(n)) for n in ndiags]

        diff = [variable_to_sympy(gp-p) for gp,p in zip(gpads, pads)]

        if d_start:bb[interface_axis] += d_start

        x_indices = []
        for i1,mi,mj,b,s,d,i2,f,xs in zip(indices1,cm,dm,bb,starts,diff,indices2,flip_axis,xshape):
            index =  b-d+FloorDiv((i1+Mod(s,mj)),mi)*mj + i2
            if f==-1:
                index = xs-1-index
            x_indices.append(index)

        out_indices = [i+m*j for i,j,m in zip(indices1,gpads,cm)]
        if c_start:out_indices[interface_axis] += c_start

        v1 = x[tuple(x_indices)]
        v2 = mat[tuple(i+m*j for i,j,m in zip(indices1,gpads,cm))+ tuple(indices2)]
        v3 = out[tuple(out_indices)]

        body = [AugAssign(v,'+' ,Mul(v2, v1))]

        # Decompose fused loop over Cartesian product of multiple ranges
        # into nested loops, each over a single range
        for i,j in zip(indices2[::-1], ranges[::-1]):
            body = [For(i,j, body)]

        body.insert(0,Assign(v, 0.0))
        body.append(Assign(v3,v))
        ranges = [Range(variable_to_sympy(i)) for i in nrows]

        # Decompose fused loop over Cartesian product of multiple ranges
        # into nested loops, each over a single range
        for i,j in zip(indices1[::-1], ranges[::-1]):
            body = [For(i,j, body)]

        if openmp:
            pragma = "#$omp for schedule(static) collapse({})".format(str(ndim))
            body   = [Comment(pragma)] + body

        nrowscopy = list(nrows).copy()
        nrows     = list(nrows)
        for dim in range(ndim):

            if nrows_extra[dim] == 0:continue

            v1 = [b-d+FloorDiv((i1+(nrows[dim] if dim==x else 0)+Mod(s,mj)),mi)*mj + i2 for x,i1,mi,mj,b,s,d,i2 in zip(range(ndim), indices1,cm,dm,bb,starts,diff,indices2)]
            v2 = [i+m*j for i,j,m in zip(indices1,gpads,cm)]

            v2[dim] += nrows[dim]

            v3 = v2

            for i,v1i in enumerate(v1):
                if flip_axis[i] == -1:
                    v1[i] = xshape[i]-1-v1[i]

            v1 = x[tuple(v1)]
            v2 = mat[tuple(v2)+ indices2]

            if c_start:v3[interface_axis] += c_start
            v3 = out[tuple(v3)]

            rows = list(nrows)
            rows[dim] = nrows_extra[dim]

            
            ranges       = [variable_to_sympy(n) for n in ndiags]
            ranges[dim] -= variable_to_sympy(indices1[dim]) + 1
            ranges       = [ind if i>=dim else ind - Max(0, variable_to_sympy(d1)+1-variable_to_sympy(r)) for i,(ind,d1,r) in enumerate(zip(ranges, indices1, nrowscopy)) ]
            ranges       = [Range(i) for i in ranges]

            for_body = [AugAssign(v, '+',Mul(v1,v2))]

            # Decompose fused loop over Cartesian product of multiple ranges
            # into nested loops, each over a single range
            for i,j in zip(indices2[::-1], ranges[::-1]):
                for_body = [For(i,j, for_body)]

            for_body.insert(0,Assign(v, 0.0))
            for_body.append(Assign(v3,v))

            ranges = [Range(variable_to_sympy(i)) for i in rows]

            # Decompose fused loop over Cartesian product of multiple ranges
            # into nested loops, each over a single range
            for i,j in zip(indices1[::-1], ranges[::-1]):
                for_body = [For(i,j, for_body)]

            if openmp:
                pragma = "#$omp for schedule(static) collapse({})".format(str(ndim))
                for_body = [Comment(pragma)] + for_body

            body += for_body

            nrows[dim] += nrows_extra[dim]

        body      = inits + body
        func_args = (mat, x, out)
        shared    = (mat, x, out)
        firstprivate = ()

        if not interface and isinstance(starts[0], Variable):
            func_args = func_args       + tuple(starts)
            firstprivate = firstprivate + tuple(starts)

        if isinstance(nrowscopy[0], Variable):
            func_args    = func_args + tuple(nrowscopy)
            firstprivate = firstprivate + tuple(nrowscopy)

        if isinstance(nrows_extra[0], Variable):
            func_args    = func_args + tuple(nrows_extra)
            firstprivate = firstprivate + tuple(nrows_extra)

        decorators = {}
        header     = None
        imports    = []
        if backend:
            if backend['name'] == 'pyccel':
                a = [String(str(i)) for i in build_pyccel_types_decorator(func_args)]
                decorators = {'types': Function('types')(*a)}
            elif backend['name'] == 'numba':
                decorators = {'njit': Function('njit')(ValuedArgument(Symbol('fastmath'), backend['fastmath']))}
            elif backend['name'] == 'pythran':
                header = build_pythran_types_header(name, func_args)

        if openmp:
            shared  = ','.join(str(a) for a in shared)
            firstprivate  = "firstprivate({})".format(','.join(str(a) for a in firstprivate)) if firstprivate else ""
            pragma1 = "#$omp parallel default(private) shared({}) {}\n".format(shared, firstprivate)
            pragma2 = "#$omp end parallel"
            body     = [Comment(pragma1)] + body + [Comment(pragma2)]
        func = FunctionDef(self.name, list(func_args), [], body, imports=imports, decorators=decorators)
        return func

    def _initialize_folder(self, folder=None):
        # ...
        if folder is None:
            basedir = os.getcwd()
            folder = PSYDAC_DEFAULT_FOLDER
            folder = os.path.join( basedir, folder )

            # ... add __init__ to all directories to be able to
            touch_init_file('__pycache__')
            for root, dirs, files in os.walk(folder):
                touch_init_file(root)
            # ...

        else:
            raise NotImplementedError('user output folder not yet available')

        folder = os.path.abspath( folder )
        mkdir_p(folder)
        # ...

        return folder

    def _generate_code(self, backend=None):

        modname = 'dependencies_{}'.format(self.tag)

        if self.comm is None or self.comm.rank == 0:

            if backend and backend['name'] == 'pyccel':
                imports  = 'from pyccel.decorators import types\n'
                imports += 'from numpy import shape'
            elif backend and backend['name'] == 'numba':
                imports  = 'from numba import njit\n'
                imports += 'from numpy import shape'
            else:
                imports = 'from numpy import shape'

            code = f'{imports}\n{pycode.pycode(self.code)}'
            write_code(modname + '.py', code, folder=self.folder)

        self._modname = modname

    def _compile(self, backend=None):

        # Make sure that code generated by process 0 is available to all others
        # Cheapest solution is a broadcast from process 0
        comm = self.comm
        if comm is not None and comm.size > 1:
            comm.bcast(0, root=0)

        module_name = self._modname
        sys.path.append(self.folder)
        importlib.invalidate_caches()
        package = importlib.import_module(module_name)
        sys.path.remove(self.folder)

        if backend and backend['name'] == 'pyccel':
            package = self._compile_pyccel(package, backend)

        self._func = getattr(package, self.name)

    def _compile_pyccel(self, mod, backend, verbose=False):

        # ... convert python to fortran using pyccel
        compiler       = backend['compiler']
        fflags         = backend['flags']
        _PYCCEL_FOLDER = backend['folder']
        accelerators   = ["openmp"] if backend["openmp"] else []

        from pyccel.epyccel import epyccel

        fmod = epyccel(mod,
                       accelerators = accelerators,
                       compiler     = compiler,
                       fflags       = fflags,
                       comm         = self.comm,
                       bcast        = True,
                       folder       = _PYCCEL_FOLDER,
                       verbose      = verbose)

        return fmod

#==============================================================================
class TransposeOperator(SplBasic):

    name_template = 'transpose_{ndim}d'
    function_dict = {1 : transpose_1d, 2 : transpose_2d, 3 : transpose_3d}

    # TODO [YG 01.04.2022]: drop support for old Pyccel versions, then remove
    args_dtype_dict = {1 : [repr('float[:,:]')]*2 + [repr('int64')]*11,
                       2 : [repr('float[:,:,:,:]')]*2 + [repr('int64')]*22,
                       3 : [repr('float[:,:,:,:,:,:]')]*2 + [repr('int64')]*33}

    def __new__(cls, ndim, comm=None, **kwargs):
        if comm is not None:
            assert isinstance(comm, MPI.Comm)
            comm_id = comm.py2f()
        else:
            comm_id = None
        return cls.__hashable_new__(ndim, comm_id, **kwargs)

    @classmethod
    @lru_cache(maxsize=32)
    def __hashable_new__(cls, ndim, comm_id=None, **kwargs):

        # If integer communicator is provided, convert it to mpi4py object
        comm = None if comm_id is None else MPI.COMM_WORLD.f2py(comm_id)

        # Generate random tag, unique for all processes in MPI communicator
        tag = random_string(8)
        if comm is not None:
            tag = comm.bcast(tag, root=0)

        # Determine name based on number of dimensions
        name = cls.name_template.format(ndim=ndim)

        # Create new instance of this class
        obj = SplBasic.__new__(cls, tag, name=name, comm=comm)

        # Initialize instance (code generation happens here)
        obj.ndim        = ndim
        backend         = dict(kwargs.pop('backend'))
        obj._code       = inspect.getsource(obj.function_dict[ndim])
        obj._args_dtype = obj.args_dtype_dict[ndim]
        obj._folder     = obj._initialize_folder()
        obj._generate_code(backend=backend)
        obj._compile(backend=backend)

        # Return instance
        return obj

    @property
    def func(self):
        return self._func

    @property
    def arguments(self):
        return self._arguments

    @property
    def code(self):
        return self._code

    @property
    def folder(self):
        return self._folder

    def _initialize_folder(self, folder=None):
        # ...
        if folder is None:
            basedir = os.getcwd()
            folder = PSYDAC_DEFAULT_FOLDER
            folder = os.path.join( basedir, folder )

            # ... add __init__ to all directories to be able to
            touch_init_file('__pycache__')
            for root, dirs, files in os.walk(folder):
                touch_init_file(root)
            # ...

        else:
            raise NotImplementedError('user output folder not yet available')

        folder = os.path.abspath( folder )
        mkdir_p(folder)
        # ...

        return folder

    def _generate_code(self, backend=None):

        modname = 'dependencies_{}'.format(self.tag)

        if self.comm is None or self.comm.rank == 0:

            dec  = ''
            code = self._code
            if backend and backend['name'] == 'pyccel':
                import pyccel
                from packaging import version
                if version.parse(pyccel.__version__) < version.parse('1.1.0'):
                    # Add @types decorator due the  minimum required Pyccel version 0.10.1
                    imports = 'from pyccel.decorators import types'
                    dec     = '@types({})'.format(','.join(self._args_dtype))
                else:
                    imports = ''
                    dec     = ''
            elif backend and backend['name'] == 'numba':
                imports = 'from numba import njit'
                dec     = '@njit(fastmath={})'.format(backend['fastmath'])
            else:
                imports = ''

            code = f'{imports}\n{dec}\n{code}'
            write_code(modname + '.py', code, folder=self.folder)

        self._modname = modname

    def _compile(self, backend=None):

        # Make sure that code generated by process 0 is available to all others
        # Cheapest solution is a broadcast from process 0
        comm = self.comm
        if comm is not None and comm.size > 1:
            comm.bcast(0, root=0)

        module_name = self._modname
        sys.path.append(self.folder)
        package = importlib.import_module( module_name )
        sys.path.remove(self.folder)

        if backend and backend['name'] == 'pyccel':
            package = self._compile_pyccel(package, backend)

        self._func = getattr(package, self.name)

    def _compile_pyccel(self, mod, backend, verbose=False):

        # ... convert python to fortran using pyccel
        compiler       = backend['compiler']
        fflags         = backend['flags']
        _PYCCEL_FOLDER = backend['folder']
        accelerators   = ["openmp"] if backend["openmp"] else []

        from pyccel.epyccel import epyccel

        fmod = epyccel(mod,
                       accelerators = accelerators,
                       compiler     = compiler,
                       fflags       = fflags,
                       comm         = self.comm,
                       bcast        = True,
                       folder       = _PYCCEL_FOLDER,
                       verbose      = verbose)

        return fmod

#==============================================================================
class InterfaceTransposeOperator(TransposeOperator):
    """ This class generates the Matrix transpose code for a StencilInterfaceMatrix.
    """

    name_template = 'interface_transpose_{ndim}d'
    function_dict = {1 : interface_transpose_1d,
                     2 : interface_transpose_2d,
                     3 : interface_transpose_3d}

    # TODO [YG 01.04.2022]: drop support for old Pyccel versions, then remove
    args_dtype_dict = {1 : [repr('float[:,:]')]*2 + [repr('int64')]*12,
                       2 : [repr('float[:,:,:,:]')]*2 + [repr('int64')]*21,
                       3 : [repr('float[:,:,:,:,:,:]')]*2 + [repr('int64')]*30}

#==============================================================================
class VectorDot(SplBasic):

    def __new__(cls, ndim, backend=None):
        tag = random_string(8)
        obj = SplBasic.__new__(cls, tag, prefix='v_dot')
        obj._ndim = ndim
        obj._backend = backend
        obj._func = obj._initialize()
        return obj

    @property
    def ndim(self):
        return self._ndim

    @property
    def func(self):
        return self._func

    @property
    def backend(self):
        return self._backend

    def _initialize(self):

        ndim = self.ndim

        indices = variables('i1:%s'%(ndim+1),'int')
        dims    = variables('n1:%s'%(ndim+1),'int')
        pads    = variables('p1:%s'%(ndim+1),'int')
        out     = variables('out','real')
        x1,x2   = variables('x1, x2','real',rank=ndim,cls=IndexedVariable)

        body = []
        ranges = [Range(p,n-p) for n,p in zip(dims,pads)]
        target = Product(*ranges)

        v1 = x1[indices]
        v2 = x2[indices]

        body = [AugAssign(out,'+' ,Mul(v1,v2))]
        body = [For(indices, target, body)]
        body.insert(0,Assign(out, 0.0))
        body.append(Return(out))

        func_args =  (x1, x2) + pads + dims

        self._imports = [Import('itertools', 'product')]

        decorators = {}
        header = None

        if self.backend['name'] == 'pyccel':
            decorators = {'types': build_pyccel_types_decorator(func_args), 'external':[]}
        elif self.backend['name'] == 'numba':
            decorators = {'jit':[]}
        elif self.backend['name'] == 'pythran':
            header = build_pythran_types_header(self.name, func_args)

        return FunctionDef(self.name, list(func_args), [], body,
                           decorators=decorators,header=header)
