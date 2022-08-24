import time

import numpy as np
import pandas as pd

from numba import njit
from numba.experimental import structref
from numba.experimental.dataframeref import DataFrameRef, DataFrameProxy
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


class TestDataFrameRefUsage(MemoryLeakMixin, TestCase):

    def setUp(self):
        super().setUp()
        self.values = np.arange(100, dtype=np.int64).reshape((10, 10))
        self.index = self.columns = [i for i in range(10)]

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

    def test_df_head(self):
        @njit
        def df_head_real(df, n=5):
            return df.head(n)

        values = self.values
        index = self.index
        columns = self.columns

        df = pd.DataFrame(values, index=index, columns=columns)
        new_df = df_head_real(df)
        print_df(new_df)


if __name__ == "__main__":
    t = TestDataFrameRefUsage()
    t.test_box()
