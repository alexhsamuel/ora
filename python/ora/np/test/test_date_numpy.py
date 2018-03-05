import datetime
import sys

import numpy as np
import pytest

import ora
from   ora import *

pytest.importorskip("ora.np")

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


arr = np.array(dates)


def test_dtype():
    assert hasattr(Date, "dtype")
    assert Date.dtype.itemsize == 4
    assert hasattr(Date16, "dtype")
    assert Date16.dtype.itemsize == 2


def test_arr_with_dtype():
    arr = np.array(dates, dtype=Date.dtype)
    assert arr.dtype is Date.dtype
    assert len(arr) == len(dates)

    for i in range(len(arr)):
        assert arr[i] == dates[i]
        assert dates[i] == arr[i]


def test_arr():
    assert arr.dtype is Date.dtype
    assert len(arr) == len(dates)

    for i in range(len(arr)):
        assert arr[i] == dates[i]
        assert dates[i] == arr[i]


def test_get_ordinal_date():
    od_arr  = ora.np.get_ordinal_date(arr)

    assert od_arr.dtype == ora.np.ORDINAL_DATE_DTYPE
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
    arr     = ora.np.date_from_ordinal_date(year, ordinal)

    assert len(arr) == len(valid_dates)
    for d0, d1 in zip(valid_dates, arr):
        assert d0 == d1


def test_date_from_ordinal_date1():
    year, ordinal = zip(*( (d.year, d.ordinal) for d in valid_dates ))
    arr = ora.np.date_from_ordinal_date(year, ordinal)

    # Should be:
    # assert (arr == np.array(valid_dates)).all()
    assert (arr == np.array(valid_dates, dtype=Date.dtype)).all()


def test_eq():
    assert (arr == arr).all()


def test_ne():
    assert not (arr != arr).any()


def test_is_valid():
    v = ora.np.is_valid(arr)
    assert (v == np.array([ d.valid for d in dates ])).all()


def test_add_shift():
    assert (arr + 1 == (
        Date.MIN + 1,
        Date.MIN + 2,
        1000/Jan/ 2,
        1001/Jan/ 1,
        2000/Jan/ 1,
        2000/Jan/ 2,
        2004/Feb/29,
        2004/Mar/ 1,
        2004/Mar/ 2,
        Date.MAX -  9999,
        Date.MAX -   999,
        Date.MAX -    99,
        Date.MAX -     9,
        Date.MAX,
        Date.INVALID,
        Date.INVALID,
        Date.INVALID,
    )).all()
    

def test_subtract_shift():
    assert (arr - 100 == (
        Date.INVALID,
        Date.INVALID,
         999/Sep/23,
        1000/Sep/22,
        1999/Sep/22,
        1999/Sep/23,
        2003/Nov/20,
        2003/Nov/21,
        2003/Nov/22,
        Date.MAX - 10100,
        Date.MAX -  1100,
        Date.MAX -   200,
        Date.MAX -   110,
        Date.MAX -   101,
        Date.MAX -   100,
        Date.INVALID,
        Date.INVALID,
    )).all()
    

def test_subtract_diff():
    dif = arr - arr
    assert (~ora.np.is_valid(arr) | (dif == 0)).all()

    sub = arr - 5
    dif = arr - sub
    assert (~ora.np.is_valid(sub) | (dif == 5)).all()


