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
    dates = [
        Date.INVALID, Date.MISSING, Date.MIN, 2016/Jun/7, 2016/Jul/4, Date.MAX,
    ]
    for i0 in range(len(dates)):
        for i1 in range(len(dates)):
            d0 = dates[i0]
            d1 = dates[i1]
            assert (i0 == i1) == (d0 == d1)
            assert (i0 != i1) == (d0 != d1)
            assert (i0 <  i1) == (d0 <  d1)
            assert (i0 <= i1) == (d0 <= d1)
            assert (i0 >  i1) == (d0 >  d1)
            assert (i0 >= i1) == (d0 >= d1)


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


def test_hash():
    dates = (
          [ 2016/Jul/10 + n for n in range(100) ]
        + [Date.INVALID, Date.MISSING])
    hashes = frozenset( hash(d) for d in dates )
    assert len(hashes) > len(dates) // 2


def test_format():
    date = 2016/Jul/10
    assert format(date, "")                     == "2016-07-10"
    assert format(date, "%Y-%m-%d")             == "2016-07-10"
    assert format(date, "%Y/%m/%d")             == "2016/07/10"
    assert format(date, "%d#%d")                == "10#10"
    assert format(date, "%G~%V~%_~W")           == "2016~27~sun"

    assert format(Date.INVALID, "%Y-%m-%d")     == "INVALID   "
    assert format(Date.INVALID, "%Y/%m/%d")     == "INVALID   "
    assert format(Date.MISSING, "%d#%d")        == "MISSI"
    assert format(Date.MISSING, "%G~%V~%^~W")   == "MISSING    "


def test_std():
    date = (2017/Dec/12).std
    assert isinstance(date, datetime.date)
    assert date.year == 2017
    assert date.month == 12
    assert date.day == 12

    with pytest.raises(ValueError):
        Date.INVALID.std
    with pytest.raises(ValueError):
        Date.MISSING.std


