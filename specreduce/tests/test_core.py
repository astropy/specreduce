import pytest

from specreduce.core import SpecreduceOperation

def test_sro_call():
    sro = SpecreduceOperation()
    with pytest.raises(NotImplementedError):
        meh = sro()

def test_sro_as_function():

    class TestSRO(SpecreduceOperation):
        def __call__(self, x):
            return x**2

    assert TestSRO.as_function(6) == 36

    class TestSRO(SpecreduceOperation):
        def __call__(self, *nargs):
            return 5

    with pytest.raises(NotImplementedError, match="There is not a way"):
        TestSRO.as_function(6) == 36