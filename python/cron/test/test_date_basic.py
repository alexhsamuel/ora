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

    with pytest.raises(ValueError):
        Date.INVALID == Date.INVALID
    with pytest.raises(ValueError):
        Date.INVALID != Date.INVALID

    with pytest.raises(ValueError):
        Date.MISSING == Date.MISSING
    with pytest.raises(ValueError):
        Date.MISSING != Date.MISSING


def test_comparison_sampled1():
    for parts in data.TEST_DATE_PARTS:
        date = Date.from_ymd(*parts)
        assert date.valid

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


def test_init_iso_date():
    assert Date("MIN") == Date.MIN
    assert Date("1973-12-03") == 1973/Dec/3
    assert Date("20160401") == 2016/Apr/1
    assert Date("MAX") == Date.MAX

    with pytest.raises(ValueError):
        Date("201641")
    with pytest.raises(ValueError):
        Date("foobar")


def test_from_iso_date():
    assert Date.from_iso_date("0001-01-01") == Date.MIN
    assert Date.from_iso_date("9999-12-31") == Date.MAX

    with pytest.raises(TypeError):
        Date.from_iso_date(None)
    with pytest.raises(TypeError):
        Date.from_iso_date(19731203)

    with pytest.raises(ValueError):
        Date.from_iso_date("1-1-1")
    with pytest.raises(ValueError):
        Date.from_iso_date("foobar")

    with pytest.raises(ValueError):
        Date.from_iso_date("1973-02-31")



