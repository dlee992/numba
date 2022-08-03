import os
import time
import unittest

import numpy as np
import pandas as pd

from numba import njit, jit
from numba.experimental import structref
from numba.experimental.dataframeproxy import DataFrameProxy
from numba.experimental.dataframeref import DataFrameRef


# os.environ['NUMBA_DEBUG_TYPEINFER'] = "1"
# os.environ['NUMBA_DUMP_SSA'] = "1"
# os.environ['NUMBA_DEBUG_JIT'] = "1"
# os.environ['NUMBA_FULL_TRACEBACKS'] = "1"
# os.environ['NUMBA_TRACE'] = "1"
os.environ['NUMBA_DUMP_OPTIMIZED'] = "1"
# os.environ['NUMBA_DUMP_FUNC_OPT'] = "1"
# os.environ['NUMBA_OPT'] = "0"
# os.environ['NUMBA_DUMP_IR'] = "1"


# we need a Rewrite pass to turn DataFrame ctor to DataFrameProxy one
structref.define_constructor(
# structref.define_proxy(
    DataFrameProxy,
    DataFrameRef,
    ["values", "index", "columns"]
)


def print_df(df):
    print(f"\nvalues={df.values}\nindex={df.index}\ncolumns={df.columns}")


class TestDataFrameRefUsage(unittest.TestCase):

    def setUp(self):
        self.values = np.arange(10, dtype=np.int64).reshape((2, 5))
        self.index = (0, 1)
        # self.columns = tuple(list(string.ascii_lowercase)[:5])
        self.columns = (3, 4, 5, 6, 7)

    def test_type_ctor(self):
        @njit
        def make_df():
            df = DataFrameProxy(values, index, columns)
            df.index = (2, 3)
            # df.columns = ('f', 'g', 'h', 'i', 'j')
            return df

        values = self.values
        index = self.index
        columns = self.columns

        df = make_df()
        print_df(df)
        # print(make_df.inspect_types())

    def test_typeof(self):
        @njit
        def typeof_test(df):
            # df.index = ("2", "3")
            return df

        values = self.values
        index = self.index
        columns = self.columns

        df = DataFrameProxy(values, index, columns)
        df = typeof_test(df)
        print_df(df)

    def test_df_eq(self):
        @njit
        def df_eq_real(df, val):
            return df.eq(val)

        values = self.values
        index = self.index
        columns = self.columns

        # use Proxy type, not Ref type
        df = DataFrameProxy(values=values, index=index, columns=columns)
        new_df = df_eq_real(df, 5)
        print_df(new_df)

    def test_df_head(self):
        @njit
        def df_head_real(df, val):
            return df.head(val)

        values = self.values
        index = self.index
        columns = self.columns

        # use Proxy type, not Ref type
        df = DataFrameProxy(values=values, index=index, columns=columns)
        new_df = df_head_real(df, 1)
        print_df(new_df)

    def test_box(self):
        @njit
        def check_box(df):
            dfp = DataFrameProxy(df.values, df.index, df.columns)
            return dfp

        # values = self.values
        # index = self.index
        # columns = self.columns

        # generate large df
        rows, cols = 10, 10
        values = np.arange(rows * cols, dtype=np.int64).reshape((rows, cols))
        index = [i+1 for i in range(rows)]
        columns = [i for i in range(cols)]

        pd_df = pd.DataFrame(values, index, columns)

        start = time.time()
        box_df = check_box(pd_df)
        print(f"njit & box time: {time.time() - start}")

    def test_get_values_inner(self):
        @njit
        def check_values(values, index, columns):
            dfp = DataFrameProxy(values, index, columns)
            values_2 = dfp.values
            return values_2

        values = self.values
        index = self.index
        columns = self.columns

        values_2 = check_values(values, index, columns)

    def test_get_index_outer(self):
        @jit
        def check_index(dfp_out):
            index_out = dfp_out.index
            return index_out

        values = self.values
        index = self.index
        columns = self.columns
        dfp = DataFrameProxy(values, index, columns)
        iout = check_index(dfp)

    def test_llvm_ir(self):
        @jit
        def add(a, b):
            return a + b

        a, b = 1, 2
        c = add(a, b)


if __name__ == "__main__":
    t = TestDataFrameRefUsage()
    t.setUp()
    t.test_box()
