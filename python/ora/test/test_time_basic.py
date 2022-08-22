import datetime
import dateutil.tz
import itertools
import numpy as np
import pytest

import ora
from   ora import Time, Time128, HiTime, NsTime, SmallTime, Unix32Time, Unix64Time
from   ora import TIME_TYPES, Date, Daytime, UTC, MIDNIGHT, TimeZone
from   ora import Jan, Jul, Nov, Dec
from   ora import to_local, from_local, now, display_time_zone, format_time
from   tools import xeq

TIME_TYPE_PAIRS = tuple(itertools.product(TIME_TYPES, TIME_TYPES))

#-------------------------------------------------------------------------------

def test_min():
    assert     Time.MIN.valid
    assert not Time.MIN.invalid
    assert not Time.MIN.missing


def test_max():
    assert     Time.MAX.valid
    assert not Time.MAX.invalid
    assert not Time.MAX.missing


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_epoch(Time):
    assert isinstance(Time.EPOCH, Time)
    assert Time.EPOCH == Time.from_offset(0)


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


def test_from_std():
    z = datetime.timezone.utc
    t = datetime.datetime(2020, 1, 22, 12, 30, 45, 250000, z)
    assert Time(t) == Time(2020, 1, 22, 12, 30, 45.25, UTC)

    z = dateutil.tz.gettz("America/New_York")
    t = datetime.datetime(2020, 1, 22, 12, 30, 45, 250000, z)
    assert Time(t) == Time(2020, 1, 22, 17, 30, 45.25, UTC)


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


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_time_min_max_str(Time):
    assert Time("MIN") == Time.MIN
    assert Time("MAX") == Time.MAX


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_add(Time):
    t0 = Time(2018, 3, 17, 12, 17, 14, UTC)
    assert t0 + 0     == t0

    assert t0 +     60      == Time(2018, 3, 17, 12, 18, 14   , UTC)
    assert t0 +  86400      == Time(2018, 3, 18, 12, 17, 14   , UTC)
    assert t0 +    -60      == Time(2018, 3, 17, 12, 16, 14   , UTC)
    assert t0 + -86400      == Time(2018, 3, 16, 12, 17, 14   , UTC)

    if Time.RESOLUTION >= 0.25:
        return

    assert t0 +      0.25   == Time(2018, 3, 17, 12, 17, 14.25, UTC)
    assert t0 +      0.5    == Time(2018, 3, 17, 12, 17, 14.5 , UTC)
    assert t0 +     -0.25   == Time(2018, 3, 17, 12, 17, 13.75, UTC)
    assert t0 +     -0.5    == Time(2018, 3, 17, 12, 17, 13.5 , UTC)


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_add_overfow(Time):
    t0 = Time(2018, 3, 17, 12, 17, 14.5, UTC)

    with pytest.raises(OverflowError): 
        t0 + 8000 * 365 * 86400

    with pytest.raises(OverflowError):
        t0 + 1e20

    with pytest.raises(OverflowError):
        t0 + -1e20

    with pytest.raises(OverflowError):
        t0 + float("inf")

    with pytest.raises(OverflowError):  # FIXME: Wrong exception?
        t0 + float("nan")


@pytest.mark.parametrize("Time0, Time1", TIME_TYPE_PAIRS)
def test_time_convert(Time0, Time1):
    assert Time1(Time0.INVALID).invalid
    assert Time1(Time0.MISSING).missing
    assert Time1(Time0(1980, 1, 1, 12, 30, 45, UTC)) == Time1(1980, 1, 1, 12, 30, 45, UTC)


def test_to_local_convert():
    t = (20200117, "12:30:45") @ UTC
    assert t == Time(2020, 1, 17, 12, 30, 45, UTC)


def test_string_convert():
    t = (20200128, "18:13:25") @ TimeZone("America/New_York")
    assert Time("2020-01-28T23:13:25+00:00") == t
    assert Time("2020-01-28T18:13:25-05:00") == t
    assert Time("2020-01-28T23:13:25Z") == t
    assert Time("2020-01-28T18:13:25R") == t
    assert Time("20200128T231325+0000") == t
    assert Time("20200128T181325-0500") == t
    assert Time("20200128T231325Z") == t
    assert Time("20200128T181325R") == t
    assert Time("2020-01-28 23:13:25+00:00") == t
    assert Time("2020-01-28 18:13:25-05:00") == t
    assert Time("2020-01-28 23:13:25Z") == t
    assert Time("2020-01-28 18:13:25R") == t


def test_out_of_range():
    with pytest.raises(ora.TimeRangeError):
        Unix32Time(2038,  1, 19,  3, 14,  6, UTC)
    with pytest.raises(ora.TimeRangeError):
        Unix32Time(2039,  1,  1,  0,  0,  0, UTC)
    with pytest.raises(ora.TimeRangeError):
        Unix32Time(9999, 12, 31, 23, 59, 59, UTC)

    with pytest.raises(ora.TimeRangeError):
        Unix32Time(1901, 12, 13, 20, 45, 51, UTC)
    with pytest.raises(ora.TimeRangeError):
        Unix32Time(1901,  1,  1,  0,  0,  0, UTC)
    with pytest.raises(ora.TimeRangeError):
        Unix32Time(   1,  1,  1,  0,  0,  0, UTC)

    with pytest.raises(ora.TimeRangeError):
        HiTime(1969, 12, 31, 23, 59, 59, UTC)
    with pytest.raises(ora.TimeRangeError):
        HiTime(2106,  2,  7,  6, 28, 17, UTC)

    with pytest.raises(ora.TimeRangeError):
        NsTime(1677,  9, 21,  0, 12, 43.145224191, UTC)
    with pytest.raises(ora.TimeRangeError):
        NsTime(2262,  4, 11, 23, 47, 16.854775806, UTC)

    with pytest.raises(ora.TimeRangeError):
        SmallTime(1969, 12, 31, 23, 59, 59, UTC)
    with pytest.raises(ora.TimeRangeError):
        SmallTime(2106,  2,  7,  6, 28, 14, UTC)


@pytest.mark.parametrize("Time", (Time, NsTime, Unix32Time, Unix64Time, SmallTime))
def test_convert_datetime64(Time):
    dt64 = np.datetime64
    assert Time(dt64("1970-01-01T00:00:00"          ,  "s")) == Time(1970,  1,  1,  0,  0,  0          , UTC)
    assert Time(dt64("2037-12-31T23:59:59"          ,  "s")) == Time(2037, 12, 31, 23, 59, 59          , UTC)
    assert Time(dt64("1970-01-01T00:00:00"          , "ms")) == Time(1970,  1,  1,  0,  0,  0          , UTC)
    assert Time(dt64("2037-12-31T23:59:59.123"      , "ms")) == Time(2037, 12, 31, 23, 59, 59.123      , UTC)
    assert Time(dt64("1970-01-01T00:00:00"          , "us")) == Time(1970,  1,  1,  0,  0,  0          , UTC)
    assert Time(dt64("2037-12-31T23:59:59.123456"   , "us")) == Time(2037, 12, 31, 23, 59, 59.123456   , UTC)
    assert Time(dt64("1970-01-01T00:00:00"          , "ns")) == Time(1970,  1,  1,  0,  0,  0          , UTC)
    assert Time(dt64("2037-12-31T23:59:59.123456789", "ns")) == Time(2037, 12, 31, 23, 59, 59.123456789, UTC)


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_convert_datetime64_nat(Time):
    assert Time(np.datetime64("NaT")).invalid
    assert Time(np.datetime64("NaT", "ns")).invalid
    assert Time(np.datetime64("NaT", "fs")).invalid


def test_convert_datetime64_range():
    with pytest.raises(OverflowError):
        SmallTime(np.datetime64("1960-01-01T00:00:00"))
    with pytest.raises(OverflowError):
        Unix32Time(np.datetime64("2038-01-19T03:14:06"))
    with pytest.raises(OverflowError):
        SmallTime(np.datetime64("1960-01-01T00:00:00", "ns"))
    with pytest.raises(OverflowError):
        Unix32Time(np.datetime64("2038-01-19T03:14:06", "ns"))


