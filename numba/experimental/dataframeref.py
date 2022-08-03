import numpy as np
import pandas as pd
from numba.core import imputils

from numba import types, TypingError
from numba.core.extending import overload_method
from numba.core.pythonapi import NativeValue, unbox, box
from numba.experimental import structref
from numba.extending import typeof_impl


@structref.register
class DataFrameRef(types.StructRef):
    pass


# almost empty class body, since do not use it in interpreter mode
class DataFrameProxy(structref.StructRefProxy):
    def __new__(cls, values, index, columns):
        return structref.StructRefProxy.__new__(cls, values, index, columns)


@typeof_impl.register(pd.DataFrame)
def _typeof_dataframe(obj, c):
    # FIXME: numpy string array should be special handled
    values_ty = typeof_impl(obj.values, c)
    index_ty = typeof_impl(obj.index.to_numpy(), c)
    columns_ty = typeof_impl(obj.columns.to_numpy(), c)
    fields = [('values', values_ty), ('index', index_ty), ('columns', columns_ty)]
    return DataFrameRef(fields)


def _unbox_attr_to_tuple(obj, attr_name, c):
    py_attr = c.pyapi.object_getattr_string(obj, attr_name)
    py_attr_arr = c.pyapi.call_method(py_attr, "to_numpy")
    c.pyapi.decref(py_attr)

    return py_attr_arr


@unbox(DataFrameRef)
def unbox_dataframe(typ, obj, c):
    """
    python state: py_df (obj)  -->  py_df_proxy
    ------------------------------------∨∨∨------------------------
    llvm state:                     ll_df_proxy  --> ll value (out)
    """
    py_df = obj

    py_values = c.pyapi.object_getattr_string(py_df, "values")
    py_index = _unbox_attr_to_tuple(py_df, "index", c)
    py_columns = _unbox_attr_to_tuple(py_df, "columns", c)

    # TODO: directly call DFP, rather than insert a module
    dfp_name = c.context.insert_const_string(
        c.builder.module, "numba.experimental.dataframeref")
    dfp_mod = c.pyapi.import_module_noblock(dfp_name)
    py_df_proxy = c.pyapi.call_method(
        dfp_mod,
        "DataFrameProxy",
        (py_values, py_index, py_columns),
    )

    mi_obj = c.pyapi.object_getattr_string(py_df_proxy, "_meminfo")

    mip_type = types.MemInfoPointer(types.voidptr)

    mi = c.unbox(mip_type, mi_obj).value

    utils = structref._Utils(c.context, c.builder, typ)
    struct_ref = utils.new_struct_ref(mi)
    out = struct_ref._getvalue()

    c.pyapi.decref(py_df_proxy)
    c.pyapi.decref(py_values)
    c.pyapi.decref(py_index)
    c.pyapi.decref(py_columns)

    c.pyapi.decref(dfp_mod)
    c.pyapi.decref(mi_obj)

    return NativeValue(out)


def box_struct_ref(typ, val, c):
    """
    Convert a raw pointer to a Python int.
    """
    utils = structref._Utils(c.context, c.builder, typ)
    struct_ref = utils.get_struct_ref(val)
    meminfo = struct_ref.meminfo

    mip_type = types.MemInfoPointer(types.voidptr)
    boxed_meminfo = c.box(mip_type, meminfo)

    obj_ctor = DataFrameProxy._numba_box_
    ctor_pyfunc = c.pyapi.unserialize(c.pyapi.serialize_object(obj_ctor))
    ty_pyobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))

    res = c.pyapi.call_function_objargs(ctor_pyfunc,
        [ty_pyobj, boxed_meminfo], )

    c.pyapi.decref(ctor_pyfunc)
    c.pyapi.decref(ty_pyobj)
    c.pyapi.decref(boxed_meminfo)

    return res


def _get_data_struct_attr(c, typ, dataval, attr):
    ret = getattr(dataval, attr)
    field_typ = typ.field_dict[attr]
    return imputils.impl_ret_borrowed(c.context, c.builder, field_typ, ret)


@box(DataFrameRef)
def box_dataframe(typ, val, c):
    """
    python state: py_df  <--  (_, _, _)
    -----------------------------^^^------------------------
    llvm state:               (_, _, _)  <-- ll_df_proxy  <-- ll value (out)
    """
    # dispatch for two usages: one for __new__.<locals>.ctor, one for pd.DF
    env_name_sub = "14StructRefProxy"
    if env_name_sub in c.env_manager.env.env_name:
        return box_struct_ref(typ, val, c)

    utils = structref._Utils(c.context, c.builder, typ)
    data_struct = utils.get_data_struct(val)

    values = _get_data_struct_attr(c, typ, data_struct, "values")
    index = _get_data_struct_attr(c, typ, data_struct, "index")
    columns = _get_data_struct_attr(c, typ, data_struct, "columns")

    py_values = c.box(typ.field_dict["values"], values)
    py_index = c.box(typ.field_dict["index"], index)
    py_columns = c.box(typ.field_dict["columns"], columns)

    pandas_mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pandas_mod = c.pyapi.import_module_noblock(pandas_mod_name)

    py_df = c.pyapi.call_method(
        pandas_mod,
        "DataFrame",
        (py_values, py_index, py_columns),
    )

    c.pyapi.decref(py_values)
    c.pyapi.decref(py_index)
    c.pyapi.decref(py_columns)
    c.pyapi.decref(pandas_mod)

    return py_df


@overload_method(DataFrameRef, "eq")
def ol_eq(self, val):
    if not isinstance(self, DataFrameRef) and not isinstance(val, types.Integer):
        raise TypingError("unsupported type input")

    def impl(self, val):
        # get length and width
        values = self.values
        rows, cols = values.shape[0], values.shape[1]
        # iterate and assign
        new_values = np.full((rows, cols), True)
        for i in range(rows):
            for j in range(cols):
                new_values[i, j] = values[i, j] >= val
        new_df = DataFrameProxy(new_values, self.index, self.columns)
        return new_df

    return impl


@overload_method(DataFrameRef, "head")
def ol_head(self, n):
    if not isinstance(self, DataFrameRef) and not isinstance(n, types.Integer):
        raise TypingError("unsupported type input")

    def impl(self, n):
        new_values = self.values[:n, :]
        new_df = DataFrameProxy(new_values, self.index, self.columns)
        return new_df

    return impl


@overload_method(DataFrameRef, "apply")
def ol_apply(self, args):
    pass
