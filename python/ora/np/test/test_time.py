import numpy as np
import ora
from   ora import now, UTC
import pytest

pytest.importorskip("ora.np")

#-------------------------------------------------------------------------------

def get_array(Time):
    """
    Array of times for testing.
    """
    arr = np.array([
        Time.MIN,
        Time(1973, 12,  3, 10, 32, 15.5, UTC),
        Time(1970,  1,  1,  0,  0,  0  , UTC),
        Time(1999, 12, 31, 23, 59, 59  , UTC),
        Time(2000,  1,  1,  0,  0,  0  , UTC),
        Time(2018,  3, 17, 14,  7, 21  , UTC),
        Time.MAX,
        Time.INVALID,
        Time.MISSING,
    ])
    assert arr.dtype is Time.dtype
    return arr


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_array(Time):
    arr = np.array([ now(Time) for _ in range(8) ])
    assert arr.dtype is Time.dtype
    assert arr.shape == (8, )


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_array_zeros(Time):
    arr = np.zeros(8, dtype=Time)
    assert arr.dtype is Time.dtype
    assert (arr == Time.from_offset(0)).all()


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_setitem(Time):
    arr = np.full(8, Time.INVALID)
    assert (arr == Time.INVALID).all()
    arr[3] = Time.MISSING
    assert not (arr == Time.INVALID).all()
    arr[2 : 4] = now(Time)
    assert arr[3] == arr[2]


def test_offset_dtype():
    assert ora.Time.offset_dtype == np.dtype("uint64")
    assert ora.Unix32Time.offset_dtype == np.dtype("int32")


OFFSETS = (
    -5e9, -60, -1, -0.25, 
    0, 
    0.015625, 0.25, 1, 60, 86400, 4e7, 5e9, 
    float("inf"), float("nan")
)

@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_time_add(Time):
    arr0 = get_array(Time)

    for offset in OFFSETS:
        if offset < Time.RESOLUTION:
            continue

        arr1 = arr0 + offset
        arr2 = offset + arr0
        for t0, t1, t2 in zip(arr0, arr1, arr2):
            assert t1 == t2  # commutativity
            if t0.valid:
                try:
                    assert t1 == t0 + offset
                except OverflowError:
                    assert t1 == Time.INVALID
            else:
                assert t1 == Time.INVALID


