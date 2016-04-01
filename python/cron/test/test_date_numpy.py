import datetime

import numpy as np
import pytest

import cron
from   cron import *
import cron.numpy

#-------------------------------------------------------------------------------

valid_dates = (
    Date.MIN,
    Date.MIN + 1,
    1000/Jan/ 1,
    1000/Dec/31,
    1999/Dec/31,
    2000/Jan/ 1,
    2004/Feb/28,
    2004/Feb/29,
    2004/Mar/ 1,
    Date.MAX - 10000,
    Date.MAX -  1000,
    Date.MAX -   100,
    Date.MAX -    10,
    Date.MAX -     1,
    Date.MAX,
)


dates = valid_dates + (
    Date.MISSING,
    Date.INVALID,
)


def test_dtype():
    assert hasattr(Date, "dtype")
    assert Date.dtype.itemsize == 4
    assert hasattr(Date16, "dtype")
    assert Date16.dtype.itemsize == 2


def test_arr():
    arr = np.array(dates, dtype=Date.dtype)
    assert len(arr) == len(dates)
    for i in range(len(arr)):
        assert arr[i].is_same(dates[i])
        assert dates[i].is_same(arr[i])


def test_get_ordinal_date():
    arr = np.array(dates, dtype=Date.dtype)
    od_arr = cron.numpy.get_ordinal_date(arr)
    assert od_arr.dtype == cron.numpy.ORDINAL_DATE_DTYPE
    assert od_arr.dtype.names == ("year", "ordinal", )
    
    for d, (y, o) in zip(dates, od_arr):
        if d.valid:
            assert y == d.year
            assert o == d.ordinal
        else:
            assert y == cron.YEAR_INVALID
            assert o == cron.ORDINAL_INVALID



