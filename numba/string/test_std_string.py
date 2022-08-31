import os

from numba.core.decorators import njit
from numba.tests.support import MemoryLeakMixin, TestCase


os.environ['NUMBA_DEBUG_TYPEINFER'] = "1"
# os.environ['NUMBA_DUMP_SSA'] = "1"
# os.environ['NUMBA_DEBUG_NRT'] = "1"
# os.environ['NUMBA_NRT_STATS'] = "1"
# os.environ['NUMBA_FULL_TRACEBACKS'] = "1"
# os.environ['NUMBA_TRACE'] = "1"
os.environ['NUMBA_DUMP_OPTIMIZED'] = "1"


class TestStdString(MemoryLeakMixin, TestCase):

    def setUp(self):
        super(TestStdString, self).setUp()

    def test_box(self):
        @njit
        def box(val):
            return val

        python_str = "abcde"
        ret = box(python_str)
        print(ret)

    def test_concat(self):
        @njit
        def concat(left, right):
            return left + right

        left_str, right_str = "abc", "def"
        ret = concat(left_str, right_str)
        print(ret)


if __name__ == '__main__':
    pass
