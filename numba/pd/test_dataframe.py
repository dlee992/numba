import os
import time

import numpy as np
import pandas as pd

from numba import njit
from numba.experimental import structref
from numba.pd.dataframe import DataFrameRef, DataFrameProxy
from numba.tests.support import MemoryLeakMixin, TestCase

# os.environ['NUMBA_DEBUG_TYPEINFER'] = "1"
# os.environ['NUMBA_DUMP_SSA'] = "1"
# os.environ['NUMBA_DEBUG_NRT'] = "1"
# os.environ['NUMBA_NRT_STATS'] = "1"
# os.environ['NUMBA_FULL_TRACEBACKS'] = "1"
# os.environ['NUMBA_TRACE'] = "1"
# os.environ['NUMBA_DUMP_OPTIMIZED'] = "1"


# we need a Rewrite pass to turn DataFrame ctor to DataFrameProxy one
structref.define_constructor(
    DataFrameProxy,
    DataFrameRef,
    ["values", "index", "columns"]
)


def print_df(df):
    print(f"\nvalues={df.values}\nindex={df.index}\ncolumns={df.columns}")


class TestDataFrame(MemoryLeakMixin, TestCase):

    def setUp(self):
        super().setUp()
        self.values = np.arange(25, dtype=np.int64).reshape((5, 5))
        self.index = self.columns = [i for i in range(5)]

    def test_box(self):
        @njit
        def check_box(df):
            return df

        # generate large df
        rows, cols = 10, 10
        values = np.arange(rows * cols, dtype=np.int64).reshape((rows, cols))
        index = [i+1 for i in range(rows)]
        columns = [i+1 for i in range(cols)]

        pd_df = pd.DataFrame(values, index=index, columns=columns)

        start = time.time()
        box_df = check_box(pd_df)
        print(box_df)
        print(f"njit & box time: {time.time() - start}")

    def test_head(self):
        @njit
        def head(df, n=5):
            return df.head(n)

        values = self.values
        index = self.index
        columns = self.columns

        df = pd.DataFrame(values, index=index, columns=columns)
        new_df = head(df)
        print_df(new_df)

    def test_transform(self):
        @njit
        def func(x):
            return x+1

        @njit
        def transform(df, func, *args):
            return df.transform(func, *args)

        values = self.values
        index = self.index
        columns = self.columns

        df = pd.DataFrame(values, index=index, columns=columns)
        # print(df)
        # old_df = df.transform(func, d=1)
        # print(old_df)
        new_df = transform(df, func)
        print(new_df)

    def test_merge(self):
        @njit
        def df_merge(left, right, how, on):
            return left.merge(right, how=how, on=on)

        values, index, columns = self.values, self.index, self.columns

        left = pd.DataFrame(values, index=index, columns=columns)
        right = pd.DataFrame(values, index=index, columns=columns)

        how, on = "left", [1, 3]
        merged = df_merge(left, right, how, on)
        expected = left.merge(right, on=on, how=how)
        print(expected)
        print(merged)


if __name__ == "__main__":
    t = TestDataFrame()
    t.test_box()
