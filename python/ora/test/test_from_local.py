import datetime
import pytz

import pytest

import ora
from   ora import Time, NsecTime, Date, Daytime, from_local, to_local, UTC
from   ora import *
import data

#-------------------------------------------------------------------------------

def test_utc0():
    t = from_local((1973/Dec/3, 7200.125), UTC)
    # FIXME
    assert(str(t) == "1973-12-03T02:00:00.12500000+00:00")


def test_nsec_time():
    # FIXME
    t = from_local((1900/Jan/1, 0), UTC, Time=NsecTime)
    assert str(t) == "1900-01-01T00:00:00.0000000000+00:00"

    t = from_local((2444/May/29, Daytime(1, 53, 3.999999997)), UTC, Time=NsecTime)
    assert str(t) == "2444-05-29T01:53:03.9999999972+00:00"

    with pytest.raises(OverflowError):
        from_local((2444/May/29, Daytime(1, 53, 5)), UTC, Time=NsecTime)

    t = from_local((1973/Dec/3, 7200.125), UTC, Time=NsecTime)
    assert str(t) == "1973-12-03T02:00:00.1250000000+00:00"


    
def test_compare_datetime():
    # Note: pytz time zones don't work correctly before 1901-12-13T15:45:42Z, 
    # which is INT_MIN seconds before the UNIX epoch.
    for yr, mo, da, ho, mi, se, tz in (
      # (1880,  1,  1, 12,  0,  0, "US/Eastern"),
      # (1883, 11, 18, 12,  0,  0, "US/Eastern"),
      # (1883, 11, 18, 12,  5,  0, "US/Eastern"),
      # (1883, 11, 18, 13,  0,  0, "US/Eastern"),
      # (1883, 11, 19,  0,  0,  0, "US/Eastern"),
        # First EST to EDT transition.
        (1918,  3, 18,  1, 59, 59, "US/Eastern"),
        (1918,  3, 18,  3,  0,  0, "US/Eastern"),
        # First EDT to EST transition.
        (1918, 10, 27,  0, 59, 59, "US/Eastern"),
        (1918, 10, 27,  1,  0,  0, "US/Eastern"),

        (1969,  7, 20, 15, 18,  4, "US/Central"),

        (2016,  3, 13,  1, 59, 59, "US/Eastern"),
        (2016,  3, 13,  3,  0,  0, "US/Eastern"),
        (2016, 11,  6,  0, 59, 59, "US/Eastern"),
        (2016, 11,  6,  1,  0,  0, "US/Eastern"),
            
    ):
        # Build localized times from parts, then convert to UTC.
        dt = datetime.datetime(yr, mo, da, ho, mi, se)
        # For ambiguous local times, our default is to use the first time.
        # These amibguous cases occur when coming off DST, so we tell localize()
        # to return the DST time rather than the regular time.
        dt = pytz.timezone(tz).localize(dt, is_dst=True)
        dt = dt.astimezone(pytz.UTC)
        
        t = from_local((Date(yr, mo, da), Daytime(ho, mi, se)), tz)
        p = to_local(t, UTC)

        assert p.date.year      == dt.year
        assert p.date.month     == dt.month
        assert p.date.day       == dt.day
        assert p.daytime.hour   == dt.hour
        assert p.daytime.minute == dt.minute
        assert p.daytime.second == dt.second + 1e-6 * dt.microsecond


def test_first():
    tz = pytz.timezone("US/Eastern")

    dt = datetime.datetime(2016, 11, 6, 1, 0, 0)
    dt = tz.localize(dt, is_dst=True).astimezone(pytz.UTC)
    t = Date(2016, 11, 6), Daytime(1, 0, 0)
    t = to_local(from_local(t, tz, first=True), UTC)
    assert t.date.year      == dt.year
    assert t.date.month     == dt.month
    assert t.date.day       == dt.day
    assert t.daytime.hour   == dt.hour
    assert t.daytime.minute == dt.minute

    dt = datetime.datetime(2016, 11, 6, 1, 0, 0)
    dt = tz.localize(dt, is_dst=False).astimezone(pytz.UTC)
    t = Date(2016, 11, 6), Daytime(1, 0, 0)
    t = to_local(from_local(t, tz, first=False), UTC)
    assert t.date.year      == dt.year
    assert t.date.month     == dt.month
    assert t.date.day       == dt.day
    assert t.daytime.hour   == dt.hour
    assert t.daytime.minute == dt.minute


