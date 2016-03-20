import datetime
import pytz

import pytest

import cron
from   cron import *
import data

#-------------------------------------------------------------------------------

def test_utc0():
    t = from_local((1973/Dec/3, 7200.125), UTC)
    # FIXME
    assert(str(t) == "1973-12-03T02:00:00.1250000Z")


def test_nsec_time():
    # FIXME
    t = from_local((1900/Jan/1, 0), UTC, Time=NsecTime)
    assert str(t) == "1900-01-01T00:00:00.000000000Z"

    t = from_local((2444/May/29, Daytime.from_parts(1, 53, 3.999999997)), UTC, Time=NsecTime)
    assert str(t) == "2444-05-29T01:53:03.999999997Z"

    t = from_local((2444/May/29, Daytime.from_parts(1, 53, 5)), UTC, Time=NsecTime)
    assert str(t) == "INVALID"

    t = from_local((1973/Dec/3, 7200.125), UTC, Time=NsecTime)
    assert str(t) == "1973-12-03T02:00:00.125000000Z"


    
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
    ):
        # Build localized times from parts, then convert to UTC.
        dt = datetime.datetime(yr, mo, da, ho, mi, se)
        dt = pytz.timezone(tz).localize(dt)
        dt = dt.astimezone(pytz.UTC)
        
        t = from_local((Date(yr, mo, da), Daytime(ho, mi, se)), tz)
        p = to_local(t, UTC)

        print(format(dt, "%Y-%m-%dT%H:%M:%S.%fZ"))
        print(t)
        print()

        assert p.date.year      == dt.year
        assert p.date.month     == dt.month
        assert p.date.day       == dt.day
        assert p.daytime.hour   == dt.hour
        assert p.daytime.minute == dt.minute
        assert p.daytime.second == dt.second + 1e-6 * dt.microsecond


