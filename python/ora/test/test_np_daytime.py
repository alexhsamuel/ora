import datetime
import itertools
import numpy as np
from   numpy.testing import assert_array_equal
import pytest

import ora
from   ora import DAYTIME_TYPES

DAYTIME_TYPE_PAIRS = tuple(itertools.product(DAYTIME_TYPES, DAYTIME_TYPES))
NP_VERSION = tuple(map(int, np.__version__.split(".")))

#-------------------------------------------------------------------------------

def get_array(Daytime):
    return np.array(
        [Daytime(0, 0), Daytime(14,31, 25), Daytime.INVALID, Daytime.MISSING])


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_daytime_array(Daytime):
    arr = get_array(Daytime)
    assert arr.dtype == Daytime.dtype


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_is_valid(Daytime):
    arr = get_array(Daytime)
    assert (ora.np.is_valid(arr).astype(int) == [1, 1, 0, 0]).all()


@pytest.mark.skipif(
    NP_VERSION < (1, 21, 0),
    reason="no TypeError in conversions before NumPy 1.21"
)
def test_convert_typeerror():
    with pytest.raises(TypeError):
        np.array([ora.Date(2019, 4, 16)], dtype=ora.Daytime)
    with pytest.raises(TypeError):
        np.array([ora.now()], dtype=ora.Daytime)


def test_convert_invalid():
    a = np.array([
        "",
        None,
        "2019-04-16",
        "2019-04-16T12:30:45+00:00",
        "12:3",
        "00:00:99",
        "0:61:00",
        "25:12:15",
    ], dtype=ora.Daytime)
    assert (a == ora.Daytime.INVALID).all()


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_to_offset(Daytime):
    daytimes = get_array(Daytime)
    offsets = ora.np.to_offset(daytimes)

    for y, o in zip(daytimes, offsets):
        assert not y.valid or o == y.offset


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_daytime_from_offset(Daytime):
    y = get_array(Daytime)
    o = ora.np.to_offset(y)

    a = ora.np.daytime_from_offset(o, Daytime=Daytime)
    assert a.dtype is Daytime.dtype
    #$ MISSING converted to INVALID in the round-trip.
    assert (((y == Daytime.MISSING) & (a == Daytime.INVALID)) | (y == a)).all()


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_daytime_from_hms(Daytime):
    h = np.array([0, 12, 2, 16, 6, 23, -1, 0, 0, 255])
    m = np.array([0, 30, 24, 17, 10, 59, 0, 60, 0, 255])
    s = np.array([0, 45, 5.75, 26.5, 47.25, 59.75, 0, 0, 60, np.nan])
    y = ora.np.daytime_from_hms(h, m, s)
    assert_array_equal(y, [
        Daytime( 0,  0,  0   ),
        Daytime(12, 30, 45   ),
        Daytime( 2, 24,  5.75),
        Daytime(16, 17, 26.5 ),
        Daytime( 6, 10, 47.25),
        Daytime(23, 59, 59.75),
        Daytime.INVALID,
        Daytime.INVALID,
        Daytime.INVALID,
        Daytime.INVALID,
    ])


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_daytime_from_ssm(Daytime):
    y = ora.np.daytime_from_ssm([
            0   ,
        45045.0 ,
         8645.75,
        58646.5 ,
        22247.25,
        86399.75,
        np.nan,
    ])
    assert_array_equal(y, [
        Daytime( 0,  0,  0   ),
        Daytime(12, 30, 45   ),
        Daytime( 2, 24,  5.75),
        Daytime(16, 17, 26.5 ),
        Daytime( 6, 10, 47.25),
        Daytime(23, 59, 59.75),
        Daytime.INVALID,
    ])


@pytest.mark.parametrize("Daytime0, Daytime1", DAYTIME_TYPE_PAIRS)
def test_cast(Daytime0, Daytime1):
    arr0 = get_array(Daytime0)
    arr1 = arr0.astype(Daytime1)

    for d0, d1 in zip(arr0, arr1):
        assert d1 == d0


@pytest.mark.parametrize("Daytime0, Daytime1", DAYTIME_TYPE_PAIRS)
def test_cast_roundtrip(Daytime0, Daytime1):
    """
    Tests that roundtrip casts work.
    """
    arr0 = get_array(Daytime0)
    arr2 = arr0.astype(Daytime1).astype(Daytime0)

    # Daytimes not representable in Daytime1 are converted to INVALID.
    assert ((arr2 == Daytime0.INVALID) | (arr2 == arr0)).all()
    assert (arr2 != Daytime0.INVALID).any()
    assert (arr2[arr0 == Daytime0.MISSING] == Daytime0.MISSING).all()


@pytest.mark.parametrize("Daytime0, Daytime1", DAYTIME_TYPE_PAIRS)
def test_compare(Daytime0, Daytime1):
    """
    Tests that comparisons between `Daytime0` and `Daytime1` work.
    """
    arr0 = np.array([
        Daytime0(0, 0, 1), Daytime0(14, 31, 25), Daytime0(23, 59, 58),
        Daytime0.INVALID, Daytime0.MISSING,
    ])
    arr1 = arr0.astype(Daytime1)
    assert (arr0 == arr1).all()
    assert (arr0 <= arr1).all()
    assert (arr0 >= arr1).all()

    arr0 = np.array([
        Daytime0(0, 0, 1), Daytime0(14, 31, 25), Daytime0(23, 59, 58),
    ])
    arr1 = arr0.astype(Daytime1)
    assert (arr0 <  arr1 + 1).all()
    assert (arr0 >  arr1 - 1).all()
    assert (arr0 != arr1 + 1).all()


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_from_object(Daytime):
    y = Daytime(12, 30, 45.5)
    arr = np.array([
        y,
        ora.Daytime(12, 30, 45.5),
        ora.Daytime32(12, 30, 45.5),
        datetime.time(12, 30, 45, 500000),
        "bogus",
        None
    ], dtype=Daytime)
    assert arr.dtype == Daytime.dtype

    assert (arr == np.array([y, y, y, y, Daytime.INVALID, Daytime.INVALID])).all()


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_get_parts(Daytime):
    arr = Daytime(12, 30, 45) + 50000 * np.arange(4)

    ha = ora.np.get_hour(arr)
    assert ha.dtype == "uint8"
    assert_array_equal(ha, [12, 2, 16, 6])

    ma = ora.np.get_minute(arr)
    assert ma.dtype == "uint8"
    assert_array_equal(ma, [30, 24, 17, 10])

    sa = ora.np.get_second(arr)
    assert sa.dtype == "float64"
    assert_array_equal(sa, [45, 5, 25, 45])


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_get_ssm(Daytime):
    y = Daytime(12, 30, 45.25)
    o = 50000 * np.arange(8)
    arr = y + o
    ssm = ora.np.get_ssm(arr)
    assert ssm.dtype == "float64"
    assert_array_equal(ssm, (y.ssm + o) % 86400)


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_get_ssm_invalid(Daytime):
    ssm = ora.np.get_ssm([Daytime.INVALID, Daytime.MISSING])
    assert np.isnan(ssm).all()


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_get_hms(Daytime):
    arr = Daytime(12, 30, 45) + 50000.75 * np.arange(4)
    arr = np.concatenate((arr, [Daytime.MISSING, Daytime.INVALID]))
    assert arr.dtype == Daytime.dtype

    hms = ora.np.get_hms(arr)
    assert hms.dtype == ora.np.HMS_DTYPE
    assert_array_equal(hms["hour"], [12, 2, 16, 6, 255, 255])
    assert_array_equal(hms["minute"], [30, 24, 17, 10, 255, 255])
    assert_array_equal(hms["second"], [45, 5.75, 26.5, 47.25, np.nan, np.nan])


