import numpy as np
import pytest

from   ora import Date, Date16
import ora.np

#-------------------------------------------------------------------------------

def to_arr(vals, dtype):
    return list(vals) if dtype is None else np.array(vals, dtype=dtype)


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
    

@pytest.mark.parametrize(
    "yt,mt,dt",
    [
        (None, None, None),
        ("int16", "int32", "int64"),
        ("uint64", "uint32", "uint16"),
    ]
)
def test_date_from_ymd_types(yt, mt, dt):
    """
    Tests that `date_from_ymd_types` works with different array dtypes.
    """
    y = to_arr([2019, 2019, 2019, 2020], yt)
    m = to_arr([   1,    4,    4,   12], mt)
    d = to_arr([   1,   30,   31,   31], dt)

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


