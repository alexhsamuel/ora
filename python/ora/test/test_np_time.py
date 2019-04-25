import numpy as np
import ora
from   ora import now, UTC
from   ora import Time, Unix32Time, Unix64Time, SmallTime, NsTime, HiTime
import ora.np
import pytest

pytest.importorskip("ora.np")

OFFSETS = (
    -5e9, -60, -1, -1.0 -0.25, 
    0, 0.0, 
    0.015625, 0.25, 1, 1.0, 60, 60.0, 86400, 4e7, 5e9, 
    float("inf"), float("nan")
)

#-------------------------------------------------------------------------------

def get_array(Time):
    """
    Array of times for testing.
    """
    arr = np.array([
        Time.MIN,
        Time(1970,  1,  1,  0,  0,  0  , UTC),
        Time(1973, 12,  3, 10, 32, 15.5, UTC),
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


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
@pytest.mark.parametrize("offset", OFFSETS)
def test_add(Time, offset):
    arr0 = get_array(Time)

    if abs(offset) < Time.RESOLUTION:
        return

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


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
@pytest.mark.parametrize("offset", OFFSETS)
def test_sub(Time, offset):
    arr0 = get_array(Time)

    if abs(offset) < Time.RESOLUTION:
        return

    arr1 = arr0 - offset
    for t0, t1 in zip(arr0, arr1):
        if t0.valid:
            try:
                assert t1 == t0 - offset
            except OverflowError:
                assert t1 == Time.INVALID
        else:
            assert t1 == Time.INVALID


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_equal(Time):
    arr = get_array(Time)
    assert ((arr == arr[2]).astype(int) == [0, 0, 1, 0, 0, 0, 0, 0, 0]).all()


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_not_equal(Time):
    arr = get_array(Time)
    assert ((arr != arr[2]).astype(int) == [1, 1, 0, 1, 1, 1, 1, 1, 1]).all()


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_less(Time):
    arr = get_array(Time)
    assert ((arr < arr[2]).astype(int) == [1, 1, 0, 0, 0, 0, 0, 1, 1]).all()


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_less_equal(Time):
    arr = get_array(Time)
    assert ((arr <= arr[2]).astype(int) == [1, 1, 1, 0, 0, 0, 0, 1, 1]).all()


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_greater(Time):
    arr = get_array(Time)
    assert ((arr > arr[2]).astype(int) == [0, 0, 0, 1, 1, 1, 1, 0, 0]).all()


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_greater_equal(Time):
    arr = get_array(Time)
    assert ((arr >= arr[2]).astype(int) == [0, 0, 1, 1, 1, 1, 1, 0, 0]).all()


@pytest.mark.parametrize("Time", ora.TIME_TYPES)
def test_is_valid(Time):
    arr = get_array(Time)
    assert (ora.np.is_valid(arr).astype(int) == [1, 1, 1, 1, 1, 1, 1, 0, 0]).all()


@pytest.mark.xfail
def test_convert_invalid():
    assert (np.array([
        "",
        None,
        "invalid",
        "2019-04-16",
        20190416,
        "12:30:45",
        "2019-04-16T12:30:45",
        "2019-04-16T12:30:45+17:00",
        "2019-04-16T12:61:45+00:00",
        ora.Date(2019, 4, 16),
        ora.Daytime(12, 30, 45),
    ], dtype=ora.Time) == ora.Time.INVALID).all()



@pytest.mark.xfail
@pytest.mark.parametrize(
    "Time0, Time1",
    [
        (SmallTime, Unix32Time),
        (Unix32Time, Unix64Time),
        (Unix64Time, Time),
        (Time, NsTime),
        (NsTime, HiTime),
    ]
)
def test_compare_types(Time0, Time1):
    arr0 = get_array(Time0)
    arr1 = arr0.astype(Time1)
    assert (arr0 == arr1).all()
    assert (arr0 <  arr1 + 1).all()
    assert (arr0 <= arr1).all()
    assert (arr0 >  arr1 - 1).all()
    assert (arr0 >= arr1).all()
    assert (arr0 != arr1 + 1).all()


def test_time_from_offset():
    offset = (1 << np.arange(30)).reshape(3, 5, 2)
    time = ora.np.time_from_offset(offset, Time=Unix32Time)
    assert time.shape == (3, 5, 2)
    assert time.dtype == Unix32Time.dtype
    assert time[0][0][0] == Unix32Time.from_offset(1)
    assert time[2][4][1] == Unix32Time.from_offset(1 << 29)


def test_time_from_offset_cast():
    offset = np.array([0, 1, 60], dtype="uint32") * Time.DENOMINATOR
    time = ora.np.time_from_offset(offset)
    assert time.shape == (3, )
    assert time.dtype == Time.dtype
    assert all(time == [
        Time(1, 1, 1, 0, 0, 0, UTC),
        Time(1, 1, 1, 0, 0, 1, UTC),
        Time(1, 1, 1, 0, 1, 0, UTC),
    ])


