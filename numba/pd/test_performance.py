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


class TestPerformance(MemoryLeakMixin, TestCase):

    def setUp(self):
        # small/median/wide/lengthy/large
        super().setUp()
        self.shapes = (
            (10, 10),
            (10**3, 10**3),
            (10**4, 10),
            (10, 10**5),
            # (10**5, 10**5),
        )
        self.exuection_times = 5

    def generate(self, shape):
        all = shape[0]*shape[1]
        values = np.arange(all, dtype=np.int64).\
            reshape((shape[0], shape[1]), order='F')
        index = [i for i in range(shape[0])]
        columns = [i for i in range(shape[1])]
        self.df = pd.DataFrame(values, index=index, columns=columns)
        self.right = pd.DataFrame(values, index=index, columns=columns)

    def warmup(self, func, lamb):
        start = time.time()
        # func.recompile()
        func(self.df, lamb)
        self.warmup_time = time.time() - start

    def average(self, func, lamb):
        start = time.time()
        for i in range(self.exuection_times):
            func(self.df, lamb)
        return (time.time() - start)/self.exuection_times

    def pprint(self):
        print(f"########################################################\n"
              f"shape={self.shape}\n"
              f"pd_average  ={self.pd_average}\n"
              f"jit_average ={self.jit_average}\n"
              f"compile time={self.warmup_time - self.jit_average}\n"
              f"********************************************************\n")

    # TODO: allow extra arguments
    def compare(self, func, lamb=None):
        self.warmup(func, lamb)
        lamb_py = lamb.py_func if hasattr(lamb, "py_func") else lamb
        self.pd_average = self.average(func.py_func, lamb_py)
        # self.pd_average = 0
        self.jit_average = self.average(func, lamb)

        self.pprint()

    def test_head(self):
        @njit
        def head(df, n=5):
            return df.head(n)

        for shape in self.shapes:
            self.shape = shape
            self.generate(shape)
            self.compare(head, 5)

    def test_transform(self):
        @njit(parallel=True)
        def lamb(x):
            return x + 1

        @njit(paralle=True)
        def transform(df, lamb, *args):
            return df.transform(lamb, *args)

        for shape in self.shapes:
            self.shape = shape
            self.generate(shape)
            self.compare(transform, lamb)


    def test_merge(self):
        @njit(parallel=True)
        def merge(left, right):
            return left.merge(right, how="left", on=[1, 3])

        for shape in self.shapes:
            self.shape = shape
            self.generate(shape)
            self.compare(merge, self.right)
