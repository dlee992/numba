from __future__ import print_function, absolute_import
import re
from llvm.core import Type, Builder, LINKAGE_INTERNAL, inline_function
from numba import typing, types
from numba.targets.base import BaseContext
from numbapro.cudadrv import nvvm


# -----------------------------------------------------------------------------
# Typing


class CUDATypingContext(typing.Context):
    def __init__(self):
        from . import cudadecl
        super(CUDATypingContext, self).__init__()
        # Load CUDA intrinsics
        for ftcls in cudadecl.INTR_FUNCS:
            self.insert_function(ftcls(self))
        for ftcls in cudadecl.INTR_ATTRS:
            self.insert_attributes(ftcls(self))
        for gv, gty in cudadecl.INTR_GLOBALS:
            self.insert_global(gv, gty)


# -----------------------------------------------------------------------------
# Implementation

VALID_CHARS = re.compile('[^a-z0-9]', re.I)


class CUDATargetContext(BaseContext):
    def init(self):
        from . import cudaimpl

        self.insert_func_defn(cudaimpl.FUNCTIONS)

    def mangler(self, name, argtypes):
        def repl(m):
            ch = m.group(0)
            return "_%X_" % ord(ch)

        qualified = name + '.' + '.'.join(str(a) for a in argtypes)
        mangled = VALID_CHARS.sub(repl, qualified)
        return mangled

    def prepare_cuda_kernel(self, func, argtypes):
        # Adapt to CUDA LLVM
        module = func.module
        func.linkage = LINKAGE_INTERNAL
        wrapper = self.generate_kernel_wrapper(func, argtypes)
        func.delete()
        del func

        nvvm.set_cuda_kernel(wrapper)
        nvvm.fix_data_layout(module)

        return wrapper

    def generate_kernel_wrapper(self, func, argtypes):
        module = func.module
        argtys = self.get_arguments(func.type.pointee)
        fnty = Type.function(Type.void(), argtys)
        wrapfn = module.add_function(fnty, name="cudaPy_" + func.name)
        builder = Builder.new(wrapfn.append_basic_block(''))

        callargs = []
        for at, av in zip(argtypes, wrapfn.args):
            av = self.get_argument_value(builder, at, av)
            callargs.append(av)

        status, _ = self.call_function(builder, func, types.void, argtypes,
                                       callargs)
        # TODO handle status

        builder.ret_void()
        del builder
        # force inline
        inline_function(status.code)

        module.verify()
        return wrapfn

    def link_dependencies(self, module, depends):
        for lib in depends:
            module.link_in(lib, preserve=True)

