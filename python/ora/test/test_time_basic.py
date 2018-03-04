import datetime
import pytest
import time

import ora
from   ora import *
from   ora import Time, Time128, HiTime, NsTime, SmallTime, Unix32Time
from   ora import Daytime, UTC, MIDNIGHT
from   ora import to_local, from_local, now, display_time_zone, format_time
import data
from   tools import xeq

#-------------------------------------------------------------------------------

def test_min():
    assert     Time.MIN.valid
    assert not Time.MIN.invalid
    assert not Time.MIN.missing


def test_max():
    assert     Time.MAX.valid
    assert not Time.MAX.invalid
    assert not Time.MAX.missing


def test_comparison():
    assert     Time.MIN     == Time.MIN
    assert     Time.MAX     != Time.MIN
    assert     Time.MIN     != Time.MAX
    assert     Time.MAX     == Time.MAX

    assert     Time.INVALID == Time.INVALID
    assert not Time.INVALID != Time.INVALID

    assert     Time.MISSING == Time.MISSING
    assert not Time.MISSING != Time.MISSING

    assert     Time.INVALID < Time.MISSING < Time.MIN < Time.MAX
    assert     Time.MAX > Time.MIN > Time.MISSING > Time.INVALID


def test_init():
    t = Time(2017, 7, 5, 19, 18, 0, "US/Eastern")
    l = to_local(t, "US/Eastern")
    assert l.date == 2017 / Jul / 5
    assert l.daytime == Daytime(19, 18, 0)


def test_init_first():
    l = Time(2018, 11, 4, 1, 30, 0, "US/Eastern", True) @ UTC
    assert l.date == 2018 / Nov / 4
    assert l.daytime == Daytime(5, 30, 0)

    l = Time(2018, 11, 4, 1, 30, 0, "US/Eastern", False) @ UTC
    assert l.date == 2018 / Nov / 4
    assert l.daytime == Daytime(6, 30, 0)


def test_zero():
    t = from_local((0, 0), UTC, Time=Unix64Time)
    p = t.get_parts(UTC)
    assert p.date.year              == 1
    assert p.date.month             == Jan
    assert p.date.day               == 1
    assert p.daytime.hour           == 0
    assert p.daytime.minute         == 0
    assert p.daytime.second         == 0
    assert p.time_zone.abbreviation == "UTC"
    assert p.time_zone.offset       == 0
    assert p.time_zone.is_dst       == False

    assert t == from_local((1/Jan/1, MIDNIGHT), UTC, Time=Time128)


def test_sub():
    t = (2016/Jul/11, Daytime(8, 32, 15)) @ UTC
    assert t - t == 0.0
    assert (t + 1) - t == 1.0
    assert xeq((t + 12345.675) - t,  12345.675, 8)
    assert xeq((t - 12345.675) - t, -12345.675, 8)
    assert xeq(t - (t + 12345.675), -12345.675, 8)
    assert xeq(t - (t - 12345.675),  12345.675, 8)


def test_hash():
    times = (
          [ Time.MIN + n for n in range(0, 270000000, 3000000) ]
        + [Time.INVALID, Time.MISSING])
    hashes = frozenset( hash(t) for t in times )
    assert len(hashes) > len(times) // 2


def test_format():
    time = (2016/Jul/11, Daytime(9, 34, 15.625)) @ UTC
    assert format(time, "%Y,%m,%d,%H,%M,%S")            == "2016,07,11,09,34,15"
    assert format(time, "%1H%M %^p")                    == "934 AM"
    assert format(time, "%%%%%H%M%.5S%%%%")             == "%%093415.62500%%"
    assert format(time, "%~_A!")                        == "mon!"

    assert format(Time.INVALID, "%Y,%m,%d,%H,%M,%S")    == "INVALID            "
    assert format(Time.INVALID, "%1H%M %^p")            == "INVALI"
    assert format(Time.MISSING, "%%%%%H%M%.5S%%%%")     == "MISSING         "
    assert format(Time.MISSING, "%~A!")                 == "MISS"


def test_format_tz():
    time = (2017/Dec/9, Daytime(15, 42, 20)) @ UTC
    assert format(time, "%Y-%m-%d %H:%M:%S@UTC") == "2017-12-09 15:42:20"
    assert format(time, "%Y-%m-%d %H:%M:%S@America/New_York") == "2017-12-09 10:42:20"
    assert "now is {:%Y-%m-%d %H:%M:%S@{}}".format(time, "America/New_York") == "now is 2017-12-09 10:42:20"


def test_format_tz_empty_utc():
    time = (2017/Dec/9, Daytime(15, 42, 20)) @ UTC
    assert format(time, "@UTC") == "2017-12-09T15:42:20+00:00"


def test_format_tz_empty_implicit():
    time = (2017/Dec/9, Daytime(15, 42, 20)) @ UTC
    with display_time_zone("America/New_York"):
        assert format(time, "@") == "2017-12-09T10:42:20-05:00"

    
def test_format_tz_empty_display():
    time = (2017/Dec/9, Daytime(15, 42, 20)) @ UTC
    with display_time_zone("America/New_York"):
        assert format(time, "@display") == "2017-12-09T10:42:20-05:00"

    
def test_format_tz_empty_in_pattern():
    time = (2017/Dec/9, Daytime(15, 42, 20)) @ UTC
    with display_time_zone("America/New_York"):
        t = format("time is now {:@}".format(time))
        assert t == "time is now 2017-12-09T10:42:20-05:00"

    
def test_format_time():
    time = (2018/Jan/1, Daytime(0, 0, 0)) @ UTC
    fmt = "%Y-%m-%d %H:%M:%S"
    assert format_time(fmt, time)                       == "2018-01-01 00:00:00"
    assert format_time(fmt, time, UTC)                  == "2018-01-01 00:00:00"
    assert format_time(fmt, time, "America/New_York")   == "2017-12-31 19:00:00"
    assert format_time(fmt, time, "Asia/Kolkata")       == "2018-01-01 05:30:00"
    tz = ora.TimeZone("Asia/Manila")
    assert format_time(fmt, time, tz)                   == "2018-01-01 08:00:00"


def test_format_method_display():
    time = (2018/Jan/1, Daytime(0, 0, 0)) @ UTC
    fmt = "%Y-%m-%d %H:%M:%S"
    with display_time_zone("America/Los_Angeles"):
        assert format_time(fmt, time, "display")        == "2017-12-31 16:00:00"


def test_format_method_system():
    time = (2018/Jan/1, Daytime(0, 0, 0)) @ UTC
    fmt = "%Y-%m-%d %H:%M:%S"
    sys = format_time(fmt, time, "system")
    try:
        stz = ora.get_system_time_zone()
    except RuntimeError:
        stz = UTC
    assert sys == format_time(fmt, time, stz)
    assert sys == format(time, fmt + "@system")


def test_from_offset():
    assert SmallTime.from_offset(SmallTime.MIN.offset) == SmallTime.MIN
    assert SmallTime.from_offset(SmallTime.MAX.offset) == SmallTime.MAX

    assert Unix32Time.from_offset(Unix32Time.MIN.offset) == Unix32Time.MIN
    assert Unix32Time.from_offset(Unix32Time.MAX.offset) == Unix32Time.MAX

    assert Time.from_offset(Time.MIN.offset) == Time.MIN
    assert Time.from_offset(Time.MAX.offset) == Time.MAX

    assert Time128.from_offset(Time128.MIN.offset) == Time128.MIN
    assert Time128.from_offset(Time128.MAX.offset) == Time128.MAX


def test_from_offset_range():
    with pytest.raises(OverflowError):
        SmallTime.from_offset(0x100000000)
    with pytest.raises(OverflowError):
        SmallTime.from_offset(0x1000000000)

    with pytest.raises(OverflowError):
        Time.from_offset(Time.MIN.offset - 1)
    with pytest.raises(OverflowError):
        Time.from_offset(Time.MAX.offset + 1)

    with pytest.raises(OverflowError):
        Time128.from_offset(Time128.MIN.offset - 1)
    with pytest.raises(OverflowError):
        Time128.from_offset(Time128.MAX.offset + 1)


def test_std():
    time = ((2016/Jul/11, Daytime(9, 34, 15.625)) @ UTC).std

    assert isinstance(time, datetime.datetime)
    assert time.year == 2016
    assert time.month == 7
    assert time.day == 11
    assert time.hour == 9
    assert time.minute == 34
    assert time.second == 15
    assert time.microsecond == 625000
    assert isinstance(time.tzinfo, datetime.tzinfo)
    assert time.tzinfo.utcoffset(time) == datetime.timedelta(0)


def test_now():
    t0 = now()
    assert isinstance(t0, ora.Time)

    t1 = now(ora.NsTime)
    assert isinstance(t1, ora.NsTime)
    assert t1 - t0 < 1

    t2 = now(ora.SmallTime)
    assert isinstance(t2, ora.SmallTime)
    assert t2 - t0 < 1


def test_now_invalid():
    with pytest.raises(TypeError):
        now(datetime.time)
    with pytest.raises(TypeError):
        now(int)


def test_HiTime():
    assert HiTime.RESOLUTION < 1e-9
    t0 = now(HiTime)
    t1 = t0 + 1e-9
    assert t1 > t0


def test_range():
    with pytest.raises(ValueError):
        Time(0, 0, 0, 0, 0, 0, UTC)
    # FIXME: Should be overflow?
    with pytest.raises(ValueError):
        Time(10000, 0, 0, 0, 0, 0, UTC)

    t0 = Time(1900, 1, 1, 0, 0, 0, UTC)
    t1 = Time(2200, 1, 1, 0, 0, 0, UTC)
    with pytest.raises(OverflowError):
        Unix32Time(t0)
    with pytest.raises(OverflowError):
        Unix32Time(t1)
    with pytest.raises(OverflowError):
        HiTime(t0)
    with pytest.raises(OverflowError):
        HiTime(t1)


@pytest.mark.xfail
def test_format_time_rounding():
    t = NsTime.from_offset(1520089388330965000)
    s = format_time("%.12S", t)
    assert s.endswith("5000000")


def test_time_from_iso():
    t = Time("2018-03-03T19:41:56-05:00")
    d, y = t @ "America/New_York"
    assert d == Date(2018, 3, 3)
    assert y == Daytime(19, 41, 56)

    t = Time("2018-03-04T00:41:56Z")
    d, y = t @ "America/New_York"
    assert d == Date(2018, 3, 3)
    assert y == Daytime(19, 41, 56)


