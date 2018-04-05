import datetime
import sys

import numpy as np
import pytest

import ora
from   ora import *

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
    year, ordinal = zip(*( (d.year, d.ordinal) for d in valid_dates ))
    arr = ora.np.date_from_ordinal_date(year, ordinal)
    assert (arr == np.array(valid_dates)).all()


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
    arr = get_array(Date)

    dif = arr - arr
    assert (~ora.np.is_valid(arr) | (dif == 0)).all()

    sub = arr - 5
    dif = arr - sub
    assert (~ora.np.is_valid(sub) | (dif == 5)).all()


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_is_valid(Date):
    arr = get_array(Date)
    iv = ora.np.is_valid(arr)
    assert iv[: -2].all() & ~iv[-2 :].any()


