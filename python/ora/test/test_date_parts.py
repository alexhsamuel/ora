import datetime

import pytest

import ora
from   ora import *

import data

#-------------------------------------------------------------------------------

def test_from_ymd():
    for year, month, day in data.TEST_DATE_PARTS:
        date = Date.from_ymd(year, month, day)
        assert date.valid
        assert not date.invalid
        assert not date.missing


def test_ymd():
    for year, month, day in data.TEST_DATE_PARTS:
        ymd = Date.from_ymd(year, month, day).ymd
        assert ymd.year   == year
        assert ymd.month  == month
        assert ymd.day    == day
        assert ymd        == (year, month, day)


def test_vs_date():
    for parts in data.TEST_DATE_PARTS:
        date = Date.from_ymd(*parts)
        ref = datetime.date(*parts)
        assert date.year    == ref.year
        assert date.month   == ref.month
        assert date.day     == ref.day


def test_vs_date_sampled():
    for date in data.sample_dates():
        ref = datetime.date.fromordinal(1 + date.datenum)
        assert date.year    == ref.year
        assert date.month   == ref.month
        assert date.day     == date.day


def test_weekday_vs_date():
    for parts in data.TEST_DATE_PARTS:
        date = Date.from_ymd(*parts)
        ref = datetime.date(*parts)
        assert date.weekday == ref.weekday()


def test_ordinal_date():
    for date, od in (
        (Date.MIN   , (   1,   1)),
        (2000/Jan/ 1, (2000,   1)),
        (2016/Jun/ 8, (2016, 160)),
        (2204/Dec/31, (2204, 366)),
        (Date.MAX   , (9999, 365)),
    ):
        assert date.ordinal_date == od
        assert date.year == od[0]
        assert date.ordinal_date.year == od[0]
        assert date.ordinal == od[1]
        assert date.ordinal_date.ordinal == od[1]

        assert Date.from_ordinal_date(od) == date
        assert Date.from_ordinal_date(*od) == date


def test_ordinal_date_invalid():
    with pytest.raises(ValueError):
        Date.INVALID.ordinal_date

    with pytest.raises(TypeError):
        Date.from_ordinal_date(None, 1)
    with pytest.raises(TypeError):
        Date.from_ordinal_date(1, None)

    with pytest.raises(ValueError):
        Date.from_ordinal_date(0, 1)
    with pytest.raises(ValueError):
        Date.from_ordinal_date(10000, 1)
    with pytest.raises(ValueError):
        Date.from_ordinal_date(1, 0)
    with pytest.raises(ValueError):
        Date.from_ordinal_date(1, 366)


def test_week_date():
    for date, wd in (
        (Date.MIN   , (   1,  1, Mon)),
        (2000/Jan/ 1, (1999, 52, Sat)),
        (2000/Jan/ 2, (1999, 52, Sun)),
        (2000/Jan/ 3, (2000,  1, Mon)),
        (2016/Jun/ 8, (2016, 23, Wed)),
        (2204/Dec/30, (2204, 52, Sun)),
        (2204/Dec/31, (2205,  1, Mon)),
        (Date.MAX   , (9999, 52, Fri)),
    ):
        assert date.week_date == wd
        assert date.week_year == wd[0]
        assert date.week_date.week_year == wd[0]
        assert date.week == wd[1]
        assert date.week_date.week == wd[1]
        assert date.weekday == wd[2]
        assert date.week_date.weekday == wd[2]

        assert Date.from_week_date(wd) == date
        assert Date.from_week_date(*wd) == date


def test_week_date_invalid():
    with pytest.raises(ValueError):
        Date.INVALID.week_date

    with pytest.raises(TypeError):
        Date.from_week_date(None, 1, Mon)
    with pytest.raises(TypeError):
        Date.from_week_date(1000, None, Mon)
    with pytest.raises(ValueError):
        Date.from_week_date(1000, 1, "Monday")

    with pytest.raises(ValueError):
        Date.from_week_date(0, 1, Mon)
    with pytest.raises(ValueError):
        Date.from_week_date(10000, 1, Mon)
    with pytest.raises(ValueError):
        Date.from_week_date(1, 0, Mon)
    with pytest.raises(ValueError):
        Date.from_week_date(1, 53, Mon)
    with pytest.raises(ValueError):
        Date.from_week_date(1, 0, 7)
    with pytest.raises(ValueError):
        Date.from_week_date(1, 0, 7)


