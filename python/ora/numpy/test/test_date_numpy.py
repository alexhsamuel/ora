import datetime
import sys

import numpy as np
import pytest

import ora
from   ora import *

pytest.importorskip("ora.numpy")

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
        assert arr[i] == dates[i]
        assert dates[i] == arr[i]


def test_get_ordinal_date():
    arr     = np.array(dates, dtype=Date.dtype)
    od_arr  = ora.numpy.get_ordinal_date(arr)

    assert od_arr.dtype == ora.numpy.ORDINAL_DATE_DTYPE
    assert od_arr.dtype.names == ("year", "ordinal", )
    
    for d, (y, o) in zip(dates, od_arr):
        if d.valid:
            assert y == d.year
            assert o == d.ordinal
        else:
            assert y == ora.YEAR_INVALID
            assert o == ora.ORDINAL_INVALID


def test_date_from_ordinal_date0():
    year    = np.array([ d.year for d in valid_dates ], dtype="int16")
    ordinal = np.array([ d.ordinal for d in valid_dates ], dtype="uint16")
    arr     = ora.numpy.date_from_ordinal_date(year, ordinal)

    assert len(arr) == len(valid_dates)
    for d0, d1 in zip(valid_dates, arr):
        assert d0 == d1


def test_date_from_ordinal_date1():
    year, ordinal = zip(*( (d.year, d.ordinal) for d in valid_dates ))
    arr = ora.numpy.date_from_ordinal_date(year, ordinal)

    # Should be:
    # assert (arr == valid_dates).all()

    for expected, date in zip(valid_dates, arr):
        assert expected == date


