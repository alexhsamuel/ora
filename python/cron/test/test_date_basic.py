import datetime

import pytest

import cron
from   cron import *

import data

#-------------------------------------------------------------------------------

def test_min():
    assert Date.MIN.valid
    assert not Date.MIN.invalid
    assert not Date.MIN.missing


def test_max():
    assert Date.MAX.valid
    assert not Date.MAX.invalid
    assert not Date.MAX.missing


def test_comparison():
    assert     Date.MIN     == Date.MIN
    assert     Date.MAX     != Date.MIN
    assert     Date.MIN     != Date.MAX
    assert     Date.MAX     == Date.MAX
    assert     Date.MIN.is_same(Date.MIN)
    assert     Date.MAX.is_same(Date.MAX)
    assert not Date.MIN.is_same(Date.MAX)
    assert not Date.MAX.is_same(Date.MIN)

    assert     Date.INVALID.is_same(Date.INVALID)
    assert not Date.INVALID == Date.INVALID
    assert not Date.INVALID != Date.INVALID

    assert     Date.MISSING.is_same(Date.MISSING)
    assert not Date.MISSING == Date.MISSING
    assert not Date.MISSING != Date.MISSING


def test_comparison_sampled1():
    for parts in data.TEST_DATE_PARTS:
        date = Date.from_parts(*parts)
        assert date.valid
        assert date.is_same(date)

        assert     date == date
        assert not date != date
        assert     date >= date
        assert not date >  date
        assert     date <= date
        assert not date <  date


def test_comparison_sampled1():
    for date0 in data.sample_dates(3881):
        for date1 in data.sample_dates(3391):
            assert date0 < date1 or date0 == date1 or date0 > date1
            assert date0 == date1 or date0 != date1
            assert date0.is_same(date1) ^ (date0 != date1)


@pytest.mark.xfail
def test_order():
    assert     Date.MAX     >= Date.MIN
    assert     Date.MIN     <= Date.MAX
    assert     Date.MIN     <  Date.MAX
    assert     Date.MAX     >  Date.MIN

    assert not Date.INVALID <= Date.INVALID
    assert not Date.INVALID <  Date.INVALID
    assert not Date.INVALID >= Date.INVALID
    assert not Date.INVALID >  Date.INVALID

    assert not Date.MISSING <= Date.MISSING
    assert not Date.MISSING <  Date.MISSING
    assert not Date.MISSING >= Date.MISSING
    assert not Date.MISSING >  Date.MISSING

