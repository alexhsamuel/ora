import numpy as np
import pytest

import ora.np
from   ora import Time, Date, Daytime, UTC

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


@pytest.mark.parametrize(
    "Time, Date, Daytime, time_zone", 
    [
        (Time, None, None, UTC),
        (Time, Date, Daytime, "America/New_York"),
        (Time, ora.Date16, ora.Daytime32, "Asia/Tokyo"),
        (ora.Unix32Time, ora.Date16, ora.Daytime32, "Asia/Tokyo"),
        (ora.Time128, ora.Date, ora.Daytime, "Europe/London"),
    ]
)
def test_to_local(Time, Date, Daytime, time_zone):
    time  = get_array(Time)
    if Date is None:
        date, daytime = ora.np.to_local(time, time_zone)
        Date = ora.Date
        Daytime = ora.Daytime
    else:
        date, daytime = ora.np.to_local(time, time_zone, Date=Date, Daytime=Daytime)
    assert date.dtype == Date.dtype
    assert daytime.dtype == Daytime.dtype

    assert date.shape == time.shape
    assert daytime.shape == time.shape

    time = time.ravel()
    date = date.ravel()
    daytime = daytime.ravel()
    for i in range(len(time)):
        try:
            d, y = ora.to_local(time[i], time_zone, Date=Date, Daytime=Daytime)
        except (ValueError, OverflowError):
            d, y = Date.INVALID, Daytime.INVALID
        assert date[i] == d
        assert daytime[i] == y


