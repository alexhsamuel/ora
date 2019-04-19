import numpy as np
import pytest

from   ora import Date
import ora.np

#-------------------------------------------------------------------------------

def to_arr(vals, dtype):
    return list(vals) if dtype is None else np.array(vals, dtype=dtype)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "yt,mt,dt",
    [
        (None, None, None),
        ("int16", "int32", "int64"),
        ("uint64", "uint32", "uint16"),
    ]
)
def test_date_from_ymd_types(yt, mt, dt):
    y = to_arr([2019, 2019, 2019, 2020], yt)
    m = to_arr([   1,    4,    4,   12], mt)
    d = to_arr([   1,   30,   31,   31], dt)

    dates = np.array([20190101, 20190430, Date.INVALID, 20201231], dtype=Date)

    res = ora.np.date_from_ymd(y, m, d)
    assert (res == dates).all()


def test_date_from_ymd_broadcast():
    dates = ora.np.date_from_ymd(2019, [1, 4], [[1, 2], [30, 31]])
    assert (dates == np.array([[20190101, 20190402], [20190130, Date.INVALID]], dtype=Date)).all()

    dates = ora.np.date_from_ymd([[2019, 2020], [2021, 1950]], 1, 31)
    assert (dates == np.array([[20190131, 20200131], [20210131, 19500131]], dtype=Date)).all()


