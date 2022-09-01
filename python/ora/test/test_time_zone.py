import datetime
import dateutil.tz
import dateutil.zoneinfo
import pytest

import ora
from   ora import *

#-------------------------------------------------------------------------------

try:
    SYSTEM_TIME_ZONE = get_system_time_zone()
except RuntimeError:
    # No system time zone set.
    SYSTEM_TIME_ZONE = None


def test_utc():
    assert UTC.name == "UTC"


def test_name():
    assert TimeZone("US/Eastern").name == "US/Eastern"


def test_us_central():
    tz = TimeZone("US/Central")
    H = 3600
    assert tz.at_local(1960/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1960/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(1969/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1969/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(1970/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1970/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(1971/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1971/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(1980/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1980/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(2000/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(2000/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(2020/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(2020/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)


def test_call_time():
    tz = TimeZone("US/Eastern")
    t0 = (2016/Jan/1, MIDNIGHT) @ tz
    t1 = (2016/Jul/1, MIDNIGHT) @ tz
    t2 = (2016/Dec/1, MIDNIGHT) @ tz

    o = tz(t0)
    assert o.offset == -18000
    assert o.abbreviation == "EST"
    assert not o.is_dst

    assert tz.at(t0) == (-18000, "EST", False)
    assert tz.at(t1) == (-14400, "EDT", True)
    assert tz.at(t2) == (-18000, "EST", False)


def test_call_local_time():
    tz = TimeZone("US/Eastern")

    o = tz((2016/Jan/1, MIDNIGHT))
    assert o.offset == -18000
    assert o.abbreviation == "EST"
    assert not o.is_dst

    assert tz(2016/Jan/1, MIDNIGHT) == (-18000, "EST", False)
    assert tz(2016/Jul/1, MIDNIGHT) == (-14400, "EDT", True)
    assert tz(2016/Dec/1, MIDNIGHT) == (-18000, "EST", False)


def test_at():
    tz = TimeZone("US/Eastern")
    t0 = (2016/Jan/1, MIDNIGHT) @ tz  # EST
    t1 = (2016/Jul/1, MIDNIGHT) @ tz  # EDT
    t2 = (2016/Dec/1, MIDNIGHT) @ tz  # back to EST

    o = tz.at(t0)
    assert o.offset == -18000
    assert o.abbreviation == "EST"
    assert not o.is_dst

    assert tz.at(t0) == (-18000, "EST", False)
    assert tz.at(t1) == (-14400, "EDT", True)
    assert tz.at(t2) == (-18000, "EST", False)


def test_at_string():
    time = now()
    lt0 = time @ TimeZone("America/New_York")
    lt1 = time @ "America/New_York"
    assert lt1 == lt0

    lt2 = time @ "US/Eastern"
    assert lt2 == lt0
    

def test_at_string_display():
    time = now()
    lt0 = time @ TimeZone("Pacific/Galapagos")

    with display_time_zone("Pacific/Galapagos"):
        lt1 = time @ "display"
    assert lt1 == lt0


def test_at_local():
    tz = TimeZone("US/Eastern")

    o = tz.at_local((2016/Jan/1, MIDNIGHT))
    assert o.offset == -18000
    assert o.abbreviation == "EST"
    assert not o.is_dst

    assert tz.at_local(2016/Jan/1, MIDNIGHT) == (-18000, "EST", False)
    assert tz.at_local(2016/Jul/1, MIDNIGHT) == (-14400, "EDT", True)
    assert tz.at_local(2016/Dec/1, MIDNIGHT) == (-18000, "EST", False)


def test_dst_transition():
    tz = TimeZone("US/Eastern")
    EST = -18000, "EST", False
    EDT = -14400, "EDT", True

    assert tz(2016/Mar/13, Daytime( 1,  0,  0)) == EST
    assert tz(2016/Mar/13, Daytime( 1, 59, 59)) == EST
    # At 2 AM, clocks spring forward one hour.
    assert tz(2016/Mar/13, Daytime( 3,  0,  0)) == EDT

    assert tz(2016/Nov/ 6, Daytime( 0,  0,  0)             ) == EDT
    assert tz(2016/Nov/ 6, Daytime( 0, 59, 59), first=True ) == EDT
    assert tz(2016/Nov/ 6, Daytime( 0, 59, 59), first=False) == EDT
    assert tz(2016/Nov/ 6, Daytime( 1,  0,  0), first=True ) == EDT
    assert tz(2016/Nov/ 6, Daytime( 1, 59, 59), first=True ) == EDT
    # At 2 AM, clocks fall back one hour.
    assert tz(2016/Nov/ 6, Daytime( 1,  0,  0), first=False) == EST
    assert tz(2016/Nov/ 6, Daytime( 1, 59, 59), first=False) == EST
    assert tz(2016/Nov/ 6, Daytime( 2,  0,  0), first=True ) == EST
    assert tz(2016/Nov/ 6, Daytime( 2,  0,  0), first=False) == EST


def test_nonexistent():
    # Should raise an exception from a nonexistent time.
    tz = TimeZone("US/Eastern")
    date = 2016/Mar/13
    daytime = Daytime(2, 30)

    with pytest.raises(ValueError):
        tz(date, daytime)
    with pytest.raises(ValueError):
        tz((date, daytime))
    with pytest.raises(ValueError):
        tz.at_local(date, daytime)
    with pytest.raises(ValueError):
        tz.at_local((date, daytime))


@pytest.mark.skipif(SYSTEM_TIME_ZONE is None, reason="no system time zone")
def test_display():
    assert isinstance(SYSTEM_TIME_ZONE, TimeZone)
    dtz = get_display_time_zone()
    assert isinstance(dtz, TimeZone)
    assert dtz == SYSTEM_TIME_ZONE

    with display_time_zone("Pacific/Galapagos"):
        assert get_system_time_zone().name == SYSTEM_TIME_ZONE.name
        dtz = get_display_time_zone()
        assert isinstance(dtz, TimeZone)
        assert dtz.name == "Pacific/Galapagos"
        assert dtz == TimeZone("Pacific/Galapagos")

    dtz = get_display_time_zone()
    assert dtz == SYSTEM_TIME_ZONE


@pytest.mark.parametrize("tz_name", ["UTC", "America/New_York", "Etc/GMT-4"])
def test_convert_dateutil_timezone(tz_name):
    du_tz = dateutil.tz.gettz(tz_name)
    tz = TimeZone(du_tz)
    assert tz.name == tz_name


@pytest.mark.parametrize("tz_name", ["UTC", "America/New_York", "US/Eastern", "Etc/GMT-4"])
def test_convert_dateutil_zonefile_timezone(tz_name):
    du_tz = dateutil.zoneinfo.get_zonefile_instance().get(tz_name)
    assert du_tz is not None
    tz = TimeZone(du_tz)
    time = now()
    assert du_tz.utcoffset(time.std).total_seconds() == tz(time).offset


def test_convert_datetime_utc():
    dt_tz = datetime.timezone.utc
    tz = ora.TimeZone(dt_tz)
    assert tz == ora.UTC


