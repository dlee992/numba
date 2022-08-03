from numba import njit, jit
from numba.experimental import structref


@jit
def dfp_get_values(self):
    return self.values


@jit
def dfp_get_index(self):
    return self.index


@jit
def dfp_get_columns(self):
    return self.columns


@jit
def dfp_set_values(self, values):
    self.values = values


@jit
def dfp_set_index(self, index):
    self.index = index


@jit
def dfp_set_columns(self, columns):
    self.columns = columns


# add `reflected` attribute, use `@reflect` to handle if True
class DataFrameProxy(structref.StructRefProxy):
    def __new__(cls, values, index, columns):
        return structref.StructRefProxy.__new__(cls, values, index, columns)

    @property
    def values(self):
        return dfp_get_values(self)

    @values.setter
    def values(self, values):
        dfp_set_values(self, values)

    @property
    def index(self):
        return dfp_get_index(self)

    @index.setter
    def index(self, index):
        dfp_set_index(self, index)

    @property
    def columns(self):
        return dfp_get_columns(self)

    @columns.setter
    def columns(self, columns):
        dfp_set_columns(self, columns)
