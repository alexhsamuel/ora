import datetime

import pytest

import cron
from   cron import *
import data
from   tools import *

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


def test_zero():
    t = from_local((0, 0), UTC)
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

    assert t == from_local((1/Jan/1, MIDNIGHT), UTC)


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
    assert format(time, "%~_W!")                        == "mon!"

    assert format(Time.INVALID, "%Y,%m,%d,%H,%M,%S")    == "INVALID            "
    assert format(Time.INVALID, "%1H%M %^p")            == "INVALI"
    assert format(Time.MISSING, "%%%%%H%M%.5S%%%%")     == "MISSING         "
    assert format(Time.MISSING, "%~W!")                 == "MISS"


def test_format_tz():
    time = (2017/Dec/9, Daytime(15, 42, 20)) @ UTC
    assert format(time, "%Y-%m-%d %H:%M:%S@UTC") == "2017-12-09 15:42:20"
    assert format(time, "%Y-%m-%d %H:%M:%S@America/New_York") == "2017-12-09 10:42:20"
    assert "now is {:%Y-%m-%d %H:%M:%S@{}}".format(time, "America/New_York") == "now is 2017-12-09 10:42:20"


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


