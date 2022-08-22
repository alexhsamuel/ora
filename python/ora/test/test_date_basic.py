import datetime
import numpy as np
import pytest

from   ora import DATE_TYPES, Date, Date16, TimeZone, now, today, UTC
from   ora import Apr, Jun, Jul, Dec

import data

#-------------------------------------------------------------------------------

@pytest.mark.parametrize("Date", DATE_TYPES)
def test_min(Date):
    assert Date.MIN.valid
    assert not Date.MIN.invalid
    assert not Date.MIN.missing


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_max(Date):
    assert Date.MAX.valid
    assert not Date.MAX.invalid
    assert not Date.MAX.missing


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_epoch(Date):
    assert isinstance(Date.EPOCH, Date)
    assert Date.EPOCH == Date.from_offset(0)


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_comparison(Date):
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


def test_comparison_sampled2():
    for date0 in data.sample_dates(3881):
        for date1 in data.sample_dates(3391):
            assert date0 < date1 or date0 == date1 or date0 > date1
            assert date0 == date1 or date0 != date1


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_order(Date):
    assert     Date.MAX     >= Date.MIN
    assert     Date.MIN     <= Date.MAX
    assert     Date.MIN     <  Date.MAX
    assert     Date.MAX     >  Date.MIN

    assert     Date.INVALID <= Date.INVALID
    assert not Date.INVALID <  Date.INVALID
    assert     Date.INVALID >= Date.INVALID
    assert not Date.INVALID >  Date.INVALID

    assert     Date.MISSING <= Date.MISSING
    assert not Date.MISSING <  Date.MISSING
    assert     Date.MISSING >= Date.MISSING
    assert not Date.MISSING >  Date.MISSING


def test_repr():
    date = 1973/Dec/3
    assert repr(date) == "Date(1973, Dec, 3)"


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
    assert format(date, "%G~%V~%_~A")           == "2016~27~sun"


def test_format_invalid():
    assert format(Date.INVALID, "%Y-%m-%d")     == "INVALID   "
    assert format(Date.INVALID, "%Y/%m/%d")     == "INVALID   "
    assert format(Date.MISSING, "%d#%d")        == "MISSI"
    assert format(Date.MISSING, "%G~%V~%^~A")   == "MISSING    "


def test_format_abbrev():
    d = Date(2018, 1, 28)
    assert format(d, "this wd is %a")  == "this wd is Sun"
    assert format(d, "this wd is %~A") == "this wd is Sun"
    assert format(d, "this mo is %b")  == "this mo is Jan"
    assert format(d, "this mo is %~B") == "this mo is Jan"


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


def test_today():
    d0 = today(UTC)
    t0 = now()
    assert isinstance(d0, Date)
    assert d0 == (t0 @ UTC).date

    d1 = today(UTC, Date16)
    t1 = now()
    assert isinstance(d1, Date16)
    assert d1 == (t1 @ UTC).date

    z = TimeZone("America/New_York")
    d2 = today(z)
    t2 = now()
    assert d2 == (t2 @ z).date


def test_today_invalid():
    with pytest.raises(TypeError):
        today(UTC, datetime.date)
    with pytest.raises(TypeError):
        today(UTC, 42)


@pytest.mark.xfail
@pytest.mark.parametrize("Date, width", [(Date, 32), (Date16, 16)])
def test_add_overflow(Date, width):
    d = Date(2018, 3, 6)
    assert d + (1 << width) == Date.INVALID
    assert d + (Date.MAX.offset - d + 1) == Date.INVALID
    assert d - (1 << width) == Date.INVALID
    assert d - (d.offset + 1) == Date.INVALID


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_convert_datetime64(Date):
    d64 = lambda n: np.datetime64(n, "D")
    assert Date(d64("1970-01-01")) == Date(1970,  1,  1)
    assert Date(d64("2022-12-31")) == Date(2022, 12, 31)
    assert Date(d64("2149-06-04")) == Date(2149,  6,  4)


@pytest.mark.parametrize("Date", DATE_TYPES)
def test_convert_datetime64_nat(Date):
    assert Date(np.datetime64("NaT")).invalid
    assert Date(np.datetime64("NaT", "D")).invalid


def test_convert_datetime64_range():
    with pytest.raises(OverflowError):
        Date16(np.datetime64("1969-12-31", "D"))
    with pytest.raises(OverflowError):
        Date16(np.datetime64("2150-01-01", "D"))


