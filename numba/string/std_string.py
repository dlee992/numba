import operator

from llvmlite import ir, binding

from numba.core import cgutils
from numba.core.types.common import Opaque
from numba.core.datamodel import models
from numba.core.imputils import lower_builtin
from numba.core.pythonapi import unbox, box, NativeValue
from numba.extending import register_model
from numba.string import _std_string


################################################################################
# type definition, registration, typeof, and boxing
################################################################################

binding.add_symbol("std_string_init", _std_string.std_string_init)
binding.add_symbol("std_string_get_cstr", _std_string.std_string_get_cstr)
binding.add_symbol("std_string_concat", _std_string.std_string_concat)


# Opaque means a meaningful raw pointer
class StdStringType(Opaque):
    def __init__(self):
        super(StdStringType, self).__init__(name='StdStringType')


std_str_type = StdStringType()
register_model(StdStringType)(models.OpaqueModel)


# define typeof_impl for str in numba.core.typing.typeof
# @typeof_impl.register(str)
# def typeof_str(val, c):
#     return std_str_type


@unbox(StdStringType)
def unbox_std_string(typ, obj, c):
    ok, buffer, size = c.pyapi.string_as_string_and_size(obj)

    fnty = ir.FunctionType(
        ir.IntType(8).as_pointer(),
        [ir.IntType(8).as_pointer(), ir.IntType(64)])
    fn = cgutils.get_or_insert_function(c.builder.module, fnty,
                                        name="std_string_init")
    ret = c.builder.call(fn, [buffer, size])

    return NativeValue(ret, is_error=c.builder.not_(ok))


@box(StdStringType)
def box_std_string(typ, val, c):
    fnty = ir.FunctionType(
        ir.IntType(8).as_pointer(),
        [ir.IntType(8).as_pointer()])
    fn = cgutils.get_or_insert_function(c.builder.module, fnty,
                                        name="std_string_get_cstr")
    c_str = c.builder.call(fn, [val])
    pystr = c.pyapi.string_from_string(c_str)
    return pystr


################################################################################
# overload string operations
################################################################################


@lower_builtin(operator.add, std_str_type, std_str_type)
def std_string_concat(context, builder, sig, args):
    fnty = ir.FunctionType(ir.IntType(8).as_pointer(),
                           [ir.IntType(8).as_pointer(),
                            ir.IntType(8).as_pointer()])
    fn = cgutils.get_or_insert_function(builder.module, fnty,
                                        name="std_string_concat")
    return builder.call(fn, args)
