import numpy as np
from   ora import *
import pytest

#-------------------------------------------------------------------------------

TIME_TYPES = (SmallTime, Unix32Time, Unix64Time, Time, NsTime, HiTime, Time128)

@pytest.mark.parametrize("Time", TIME_TYPES)
def test_array(Time):
    arr = np.array([ now(Time) for _ in range(8) ])
    assert arr.dtype is Time.dtype
    assert arr.shape == (8, )


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_array_zeros(Time):
    arr = np.zeros(8, dtype=Time)
    assert arr.dtype is Time.dtype
    assert (arr == Time.from_offset(0)).all()


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_setitem(Time):
    arr = np.full(8, Time.INVALID)
    assert (arr == Time.INVALID).all()
    arr[3] = Time.MISSING
    assert not (arr == Time.INVALID).all()
    arr[2 : 4] = now(Time)
    assert arr[3] == arr[2]


