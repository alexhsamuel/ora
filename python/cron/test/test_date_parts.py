import datetime

import pytest

import cron
from   cron import *

import data

#-------------------------------------------------------------------------------

def test_parts_attrs():
    for year, month, day in data.TEST_DATE_PARTS:
        date = Date.from_parts(year, month, day)
        assert date.valid
        assert date.year == year
        assert date.month == month
        assert date.day == day


def test_parts():
    for year, month, day in data.TEST_DATE_PARTS:
        parts = Date.from_parts(year, month, day).parts
        assert parts.year   == year
        assert parts.month  == month
        assert parts.day    == day
        assert parts[: 3]   == (year, month, day)


def test_vs_date():
    for parts in data.TEST_DATE_PARTS:
        date = Date.from_parts(*parts)
        ref = datetime.date(*parts)
        assert date.year    == ref.year
        assert date.month   == ref.month
        assert date.day     == ref.day


