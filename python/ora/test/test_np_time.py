import itertools
import numpy as np
import pytest

import ora
from   ora import now, UTC, TIME_TYPES
from   ora import Time, Unix32Time
import ora.np

NP_VERSION = tuple(map(int, np.__version__.split(".")))

TIME_TYPE_PAIRS = tuple(itertools.product(TIME_TYPES, TIME_TYPES))

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


def test_offset_dtype():
    assert ora.Time.offset_dtype == np.dtype("uint64")
    assert ora.Unix32Time.offset_dtype == np.dtype("int32")


@pytest.mark.parametrize("Time", TIME_TYPES)
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


@pytest.mark.parametrize("Time", TIME_TYPES)
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


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_equal(Time):
    arr = get_array(Time)
    assert ((arr == arr[2]).astype(int) == [0, 0, 1, 0, 0, 0, 0, 0, 0]).all()


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_not_equal(Time):
    arr = get_array(Time)
    assert ((arr != arr[2]).astype(int) == [1, 1, 0, 1, 1, 1, 1, 1, 1]).all()


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_less(Time):
    arr = get_array(Time)
    assert ((arr < arr[2]).astype(int) == [1, 1, 0, 0, 0, 0, 0, 1, 1]).all()


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_less_equal(Time):
    arr = get_array(Time)
    assert ((arr <= arr[2]).astype(int) == [1, 1, 1, 0, 0, 0, 0, 1, 1]).all()


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_greater(Time):
    arr = get_array(Time)
    assert ((arr > arr[2]).astype(int) == [0, 0, 0, 1, 1, 1, 1, 0, 0]).all()


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_greater_equal(Time):
    arr = get_array(Time)
    assert ((arr >= arr[2]).astype(int) == [0, 0, 1, 1, 1, 1, 1, 0, 0]).all()


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_is_valid(Time):
    arr = get_array(Time)
    assert (ora.np.is_valid(arr).astype(int) == [1, 1, 1, 1, 1, 1, 1, 0, 0]).all()


def test_convert_invalid():
    for obj in [
        "",
        "invalid",
        "2019-04-16",
        20190416,
        "12:30:45",
        "2019-04-16T12:30:45",
        "2019-04-16T12:30:45+Z:00",
        "2019-04-16T12:61:45+00:00",
        ora.Date(2019, 4, 16),
        ora.Daytime(12, 30, 45),
    ]:
        print(obj)
        with pytest.raises((TypeError, ValueError)):
            np.array([obj], dtype=ora.Time)


@pytest.mark.parametrize("Time0, Time1", TIME_TYPE_PAIRS)
def test_compare_types(Time0, Time1):
    arr0 = np.array([
        Time0(1970,  1,  2,  0,  0,  0  , UTC),
        Time0(1999, 12, 31, 23, 59, 59  , UTC),
        Time0(2000,  1,  1,  0,  0,  0  , UTC),
        Time0(2018,  3, 17, 14,  7, 21  , UTC),
    ])

    arr1 = arr0.astype(Time1)
    assert (arr0 == arr1).all()
    assert (arr0 <= arr1).all()
    assert (arr0 >= arr1).all()

    assert (arr0 <  arr1 + 1).all()
    assert (arr0 >  arr1 - 1).all()
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


@pytest.mark.xfail
@pytest.mark.parametrize("Time0, Time1", TIME_TYPE_PAIRS)
def test_cast(Time0, Time1):
    arr0 = get_array(Time0)
    arr1 = arr0.astype(Time1)

    for d0, d1 in zip(arr0, arr1):
        try:
            Time1(d0)
        except ora.TimeRangeError:
            assert d1 == Time1.INVALID
        else:
            assert d0 == d1 or abs(d1 - d0) <= max(d0.RESOLUTION, d1.RESOLUTION)


@pytest.mark.parametrize("Time0, Time1", TIME_TYPE_PAIRS)
def test_cast_scalar(Time0, Time1):
    scalar0 = Time0(1973, 12, 3, 12, 45, 31, UTC)
    scalar1 = Time1(scalar0)
    # FIXME: I'm not sure this is a feature...
    scalar2 = scalar0.astype(Time0)

    assert scalar1 == scalar0
    assert scalar2 == scalar0
    assert scalar2 == scalar1
    assert scalar1 == Time1(1973, 12, 3, 12, 45, 31, UTC)
    assert np.array(scalar0) == scalar0


@pytest.mark.xfail
@pytest.mark.parametrize("Time0, Time1", TIME_TYPE_PAIRS)
def test_cast_roundtrip(Time0, Time1):
    """
    Tests that roundtrip casts work.
    """
    arr0 = get_array(Time0)
    arr2 = arr0.astype(Time1).astype(Time0)

    # Times not representable in Time1 are converted to INVALID.
    assert ((arr2 == Time0.INVALID) | (arr2 == arr0)).all()
    assert (arr2 != Time0.INVALID).any()
    assert (arr2[arr0 == Time0.MISSING] == Time0.MISSING).all()


@pytest.mark.xfail(
    NP_VERSION[0] == 1 and 21 <= NP_VERSION[1] <= 22,
    reason="https://github.com/numpy/numpy/issues/21365"
)
@pytest.mark.parametrize("Time", TIME_TYPES)
@pytest.mark.parametrize("units", ["s", "ms", "us", "ns"])
def test_cast_datetime64(Time, units):
    dtype = np.dtype(f"datetime64[{units}]")

    dates = [
        "1900-01-01", "1969-07-20", "1970-01-01", "2019-05-26", "2038-01-05",
        "2149-06-04",
    ]
    dates = [ d for d in dates if (Time.MIN @ UTC).date < ora.Date(d) < (Time.MAX @ UTC).date ]
    daytimes = ["00:00:00", "20:17:00", "23:59:59"]
    if units in {"ms", "us", "ns"} and Time.RESOLUTION <= 1e-3:
        daytimes.append("01:23:45.678")
    if units in {"us", "ns"} and Time.RESOLUTION <= 1e-6:
        daytimes.append("09:01:23.456789")
    if units in {"ns"} and Time.RESOLUTION <= 1e-9:
        daytimes.append("12:34:56.789012345")

    dt64 = [ d + "T" + y for d in dates for y in daytimes ]
    dt64.append("NAT")
    dt64 = np.array(dt64, dtype=dtype)

    times = [ d + "T" + y + "Z" for d in dates for y in daytimes ]
    times.append(Time.INVALID)
    times = np.array(times, dtype=Time)

    dt64_res = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[units]

    # Convert datetime64 to Time.  Check that the result diffs are less than the
    # time resolution.
    cnv = dt64.astype(Time)
    assert ((cnv == times) | (np.abs(cnv - times) < dt64_res)).all()

    # Convert Time to datetime64.  Check that the resulting diffs are less than
    # the time resolution.
    diff = np.abs(times.astype(dtype) - dt64).astype(int) * dt64_res
    assert (diff < Time.RESOLUTION).all()


