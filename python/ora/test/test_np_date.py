import numpy as np
import pytest

import ora
from   ora import DATE_TYPES, Date, Date16

#-------------------------------------------------------------------------------

def get_array(Date):
    return np.array([
        Date.MIN,
        Date.MIN + 1,
        Date(1973,  1,  1),
        Date(1973, 12, 31),
        Date(1999, 12, 31),
        Date(2000,  1,  1),
        Date(2004,  2, 28),
        Date(2004,  2, 29),
        Date(2004,  3,  1),
        Date.MAX - 10000,
        Date.MAX -  1000,
        Date.MAX -   100,
        Date.MAX -    10,
        Date.MAX -     1,
        Date.MAX,
        Date.MISSING,
        Date.INVALID,
    ])
    

def test_dtype():
    assert Date.dtype.itemsize == 4
    assert Date16.dtype.itemsize == 2


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_arr(Date):
    arr = get_array(Date)
    assert arr.dtype is Date.dtype


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_get_ordinal_date(Date):
    arr = get_array(Date)
    od_arr = ora.np.get_ordinal_date(arr)

    assert od_arr.dtype == ora.np.ORDINAL_DATE_DTYPE
    assert od_arr.dtype.names == ("year", "ordinal", )
    
    for d, (y, o) in zip(arr, od_arr):
        if d.valid:
            assert y == d.year
            assert o == d.ordinal
        else:
            assert y == ora.YEAR_INVALID
            assert o == ora.ORDINAL_INVALID


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_date_from_ordinal_date0(Date):
    dates   = get_array(Date)[: -2]
    year    = np.array([ d.year for d in dates ], dtype="int16")
    ordinal = np.array([ d.ordinal for d in dates ], dtype="uint16")
    arr     = ora.np.date_from_ordinal_date(year, ordinal, Date=Date)

    assert len(arr) == len(dates)
    for d0, d1 in zip(dates, arr):
        assert d0 == d1


def test_date_from_ordinal_date1():
    dates = get_array(Date)
    dates = dates[ora.np.is_valid(dates)]
    year, ordinal = zip(*( (d.year, d.ordinal) for d in dates ))
    arr = ora.np.date_from_ordinal_date(year, ordinal)
    assert (arr == np.array(dates)).all()


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_eq(Date):
    arr = get_array(Date)
    assert (arr == arr).all()


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_ne(Date):
    arr = get_array(Date)
    assert not (arr != arr).any()


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_is_valid(Date):
    arr = get_array(Date)
    v = ora.np.is_valid(arr)
    assert (v == np.array([ d.valid for d in arr ])).all()


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_is_valid2(Date):
    arr = get_array(Date)
    iv = ora.np.is_valid(arr)
    assert iv[: -2].all() & ~iv[-2 :].any()


def test_add_shift():
    arr = np.array([
        Date.MIN,
        Date(2019, 4, 20),
        Date.MAX - 100,
        Date.MAX - 1,
        Date.MAX,
        Date.MISSING,
        Date.INVALID,
    ]) + 100
    assert (arr == np.array([
        Date.MIN + 100,
        Date(2019, 7, 29),
        Date.MAX,
        Date.INVALID,
        Date.INVALID,
        Date.INVALID,
        Date.INVALID,
    ])).all()
    

def test_subtract_shift():
    arr = np.array([
        Date.MIN,
        Date.MIN + 1,
        Date.MIN + 100,
        Date(2019, 4, 20),
        Date.MAX,
        Date.MISSING,
        Date.INVALID,
    ]) - 100
    assert (arr == np.array([
        Date.INVALID,
        Date.INVALID,
        Date.MIN,
        Date(2019, 1, 10),
        Date.MAX - 100,
        Date.INVALID,
        Date.INVALID,
    ])).all()
    

def test_subtract_diff():
    arr = get_array(Date)

    dif = arr - arr
    assert (~ora.np.is_valid(arr) | (dif == 0)).all()

    sub = arr - 5
    dif = arr - sub
    assert (~ora.np.is_valid(sub) | (dif == 5)).all()


@pytest.mark.xfail
def test_convert_invalid():
    assert (np.array([
        None,
        "bogus"
        "2012-02-30",
        "2013-02-29",
        "2012-02",
        20120230,
        2012023,
        201202301,
        "missing",
        "87654321",
        ora.now(),
        ora.Daytime(12, 30, 45),
    ], dtype=Date) == Date.INVALID).all()


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_to_offset(Date):
    dates = get_array(Date)
    dates = dates[ora.np.is_valid(dates)]
    offsets = ora.np.to_offset(dates)
    assert (offsets == [ d.offset for d in dates ]).all()


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_date_from_offset(Date):
    dates = get_array(Date)
    # MISSING dates won't survive the round-trip.
    dates[dates == Date.MISSING] = Date.INVALID

    offsets = ora.np.to_offset(dates)
    assert (ora.np.date_from_offset(offsets, Date=Date) == dates).all()

    offsets = offsets.astype("int64")
    assert (ora.np.date_from_offset(offsets, Date=Date) == dates).all()
    

@pytest.mark.xfail
@pytest.mark.parametrize(
    "Date0, Date1",
    [
        (Date16, Date),
    ]
)
def test_cast(Date0, Date1):
    """
    Tests that casts between `Date0` and wider `Date1` work.
    """
    arr0 = get_array(Date0)
    arr1 = arr0.astype(Date1)
    assert (arr1 == arr0).all()
    assert (arr1.astype(Date0) == arr0).all()


@pytest.mark.parametrize(
    "yt,mt,dt",
    [
        (int, int, int),
        ("int16", "int32", "int64"),
        ("uint64", "uint32", "uint16"),
    ]
)
def test_date_from_ymd_types(yt, mt, dt):
    """
    Tests that `date_from_ymd_types` works with different array dtypes.
    """
    y = np.array([2019, 2019, 2019, 2020], dtype=yt)
    m = np.array([   1,    4,    4,   12], dtype=mt)
    d = np.array([   1,   30,   31,   31], dtype=dt)

    dates = np.array([20190101, 20190430, Date.INVALID, 20201231], dtype=Date)

    res = ora.np.date_from_ymd(y, m, d)
    assert (res == dates).all()


def test_date_from_ymd_broadcast():
    """
    Tests that `date_from_ymd_broadcast` broadcasts its args.
    """
    dates = ora.np.date_from_ymd(2019, [1, 4], [[1, 2], [30, 31]])
    assert (dates == np.array([[20190101, 20190402], [20190130, Date.INVALID]], dtype=Date)).all()

    dates = ora.np.date_from_ymd([[2019, 2020], [2021, 1950]], 1, 31)
    assert (dates == np.array([[20190131, 20200131], [20210131, 19500131]], dtype=Date)).all()


@pytest.mark.xfail
@pytest.mark.parametrize(
    "Date,dtype",
    [
        (Date, int),
        (Date16, "uint32"),
        (Date16, "int32"),
    ]
)
def test_date_from_ymdi_types(Date, dtype):
    """
    Tests that `date_from_ymdi` works with different dtypes.
    """
    dates = get_array(Date)
    # MISSING won't survive the round trip, so remove it.
    dates[dates == Date.MISSING] = Date.INVALID

    # Round-trip it.
    ymdi = ora.np.get_ymdi(dates).astype(dtype)
    res = ora.np.date_from_ymdi(ymdi, Date=Date)
    assert (res == dates).all()


@pytest.mark.xfail
def test_date_from_ymdi_broadcast():
    """
    Tests that `date_from_ymdi` works for different shapes.
    """
    # Scalar.
    dates = ora.np.date_from_ymdi(20190420)
    assert dates == np.array(Date(2019, 4, 20))

    # 1D.
    dates = ora.np.date_from_ymdi([20190420, 20190430, 20190101])
    assert dates == np.array(
        [Date(2019, 4, 20), Date.INVALID, Date(2019, 1, 1)])

    # 2D.
    dates = ora.np.date_from_ymdi([[20190420, 20190430], [20190101, 19731231]])
    assert dates == np.array([
        [Date(2019, 4, 20), Date.INVALID],
        [Date(201, 1, 1), Date(1973, 12, 31)]
    ])


