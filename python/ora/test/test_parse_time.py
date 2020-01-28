import pytest

from   ora import parse_time, parse_time_iso, Time, Time128, UTC, ParseError

#-------------------------------------------------------------------------------

def test_basic():
    assert parse_time("%Y-%m-%dT%H:%M:%S%E", "2018-02-20T15:49:42-07:00") == Time(2018,  2, 20, 15, 49, 42, "US/Mountain")
    assert parse_time("%Y-%m-%dT%H:%M:%S%E", "2018-02-20T17:49:42-05:00") == Time(2018,  2, 20, 17, 49, 42, "US/Eastern")
    assert parse_time("%Y-%m-%dT%H:%M:%S%E", "2018-02-20T22:49:42+00:00") == Time(2018,  2, 20, 22, 49, 42, "UTC")

    assert parse_time("%Y-%m-%dT%H:%M:%S%E", "2018-02-20T22:49:42.125+00:00") == Time(2018,  2, 20, 22, 49, 42.125, "UTC")

    assert parse_time("%Y-%m-%dT%H:%M:%S%z", "2018-02-20T15:49:42-0700" ) == Time(2018,  2, 20, 15, 49, 42, "US/Mountain")
    assert parse_time("%Y-%m-%dT%H:%M:%S%z", "2018-02-20T17:49:42-0500" ) == Time(2018,  2, 20, 17, 49, 42, "US/Eastern")
    assert parse_time("%Y-%m-%dT%H:%M:%S%z", "2018-02-20T22:49:42+0000" ) == Time(2018,  2, 20, 22, 49, 42, "UTC")

    assert parse_time("%Y-%m-%dT%H:%M:%S%z", "2018-02-20T22:49:42.390625-0200" ) == Time(2018,  2, 21,  0, 49, 42.390625, "UTC")


def test_min():
    t = Time128(1, 1, 1, 0, 0, 0, "UTC")
    assert parse_time("%Y-%m-%d %H:%M:%S %E", "0001-01-01 00:00:00 +00:00", Time=Time128) == t
    assert parse_time("%Y-%m-%d %H:%M:%S %E", "0001-01-01 12:00:00 +12:00", Time=Time128) == t


def test_max():
    t = Time128(9999, 12, 31, 23, 59, 59, "UTC")
    assert parse_time("%Y-%m-%d %H:%M:%S %E", "9999-12-31 23:59:59 +00:00", Time=Time128) == t
    assert parse_time("%Y-%m-%d %H:%M:%S %E", "9999-12-31 11:59:59 -12:00", Time=Time128) == t


def test_tz_letter():
    assert parse_time("%Y-%m-%dT%H:%M:%S%e", "2018-02-20T15:49:42T") == Time(2018,  2, 20, 15, 49, 42, "US/Mountain")
    assert parse_time("%Y-%m-%dT%H:%M:%S%e", "2018-02-20T17:49:42R") == Time(2018,  2, 20, 17, 49, 42, "US/Eastern")
    assert parse_time("%Y-%m-%dT%H:%M:%S%e", "2018-02-20T22:49:42Z") == Time(2018,  2, 20, 22, 49, 42, "UTC")
    assert parse_time("%Y-%m-%dT%H:%M:%S%e", "2018-02-21T10:49:42M") == Time(2018,  2, 21, 10, 49, 42, "Etc/GMT-12")


def test_iso():
    assert parse_time("%i", "2018-02-20T15:49:42+00:00") == Time(2018,  2, 20, 15, 49, 42, "UTC")
    assert parse_time("%i", "2018-02-20T15:49:42-12:00") == Time(2018,  2, 20, 15, 49, 42, "Etc/GMT+12")
    assert parse_time("%i", "2018-02-20T15:49:42+05:30") == Time(2018,  2, 20, 15, 49, 42, "Asia/Kolkata")
    assert parse_time("%i", "2018-02-20T15:49:42+12:00") == Time(2018,  2, 20, 15, 49, 42, "Etc/GMT-12")

    assert parse_time("%i", "2018-02-20T15:59:59.75-05:00") == Time(2018,  2, 20, 20, 59, 59.75, UTC)


def test_iso_letter():
    assert parse_time("%T", "2018-02-20T15:49:42Z") == Time(2018,  2, 20, 15, 49, 42, "UTC")


def test_tz_name():
    assert parse_time("%Y-%m-%d %H:%M %Z", "2018-02-20 20:50 US/Eastern") == Time(2018, 2, 20, 20, 50, 0, "US/Eastern")
    assert parse_time("%Y-%m-%d %H:%M %Z", "2018-02-20 20:50 UTC") == Time(2018, 2, 20, 20, 50, 0, UTC)
    assert parse_time("%Y-%m-%d %H:%M %Z", "2018-02-20 20:50 Asia/Tokyo") == Time(2018, 2, 20, 20, 50, 0, "Asia/Tokyo")
    assert parse_time("%Y-%m-%d %H:%M %Z foo", "2018-02-20 20:50 Asia/Kolkata foo") == Time(2018, 2, 20, 20, 50, 0, "Asia/Kolkata")


def test_tz_name_invalid():
    with pytest.raises(ValueError):
        parse_time("%Y-%m-%d %H:%M %Z", "2018-02-20 20:50 America/London")
    with pytest.raises(ValueError):
        parse_time("%Y-%m-%d %H:%M [%Z]", "2018-02-20 20:50 []")


def test_default_time_zone():
    with pytest.raises(ValueError):
        parse_time("%Y-%m-%d %H:%M", "2018-02-20 20:48")

    t = parse_time("%Y-%m-%d %H:%M", "2018-02-20 20:48", time_zone=UTC)
    assert t == Time(2018, 2, 20, 20, 48, 0, UTC)

    t = parse_time("%Y-%m-%d %H:%M", "2018-02-20 20:48", time_zone="US/Eastern")
    assert t == Time(2018, 2, 20, 20, 48, 0, "America/New_York")

    t = parse_time("%Y-%m-%d %H:%M %Z", "2018-02-20 20:48 America/Los_Angeles", time_zone="US/Eastern")
    assert t == Time(2018, 2, 20, 20, 48, 0, "America/Los_Angeles")


def test_parse_time_iso():
    t = parse_time_iso("2018-02-22T04:09:11.25282922+00:00")
    assert t == Time(2018, 2, 22, 4, 9, 11.25282922, UTC)

    t = parse_time_iso("2018-02-22T04:09:11.2528+00:00")
    assert t == Time(2018, 2, 22, 4, 9, 11.2528, UTC)

    t = parse_time_iso("2018-02-22T04:09:11.+00:00")
    assert t == Time(2018, 2, 22, 4, 9, 11, UTC)

    t = parse_time_iso("2018-02-22T04:09:11+00:00")
    assert t == Time(2018, 2, 22, 4, 9, 11, UTC)

    t = parse_time_iso("2018-02-22T05:09:11+01:00")
    assert t == Time(2018, 2, 22, 4, 9, 11, UTC)

    t = parse_time_iso("2018-02-22T03:09:11-01:00")
    assert t == Time(2018, 2, 22, 4, 9, 11, UTC)

    t = parse_time_iso("2018-02-22T04:09:11Z")
    assert t == Time(2018, 2, 22, 4, 9, 11, UTC)

    t = parse_time_iso("2018-02-22t05:09:11+01:00")
    assert t == Time(2018, 2, 22, 4, 9, 11, UTC)



def test_parse_time_iso_invalid():
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22T04:09:11.25282922+0:00")
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22T04:09:11.25282922+00:0")
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22T04:09:11.25282922+0000")
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22T04:09:11.25282922*00:00")
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22T04:09:11.25282922")
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22T04:09+00:00")
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22T04:9:11.25282922+00:00")
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22T4:09:11.25282922+00:00")
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22 04:09:11.25282922 +00:00")
    with pytest.raises(ValueError):
        parse_time_iso("  2018-02-22T04:09:11.25282922+00:00")
    with pytest.raises(ValueError):
        parse_time_iso("2018-02-22T04:09:11.25282922+00:00 is now")
    with pytest.raises(ParseError):
        parse_time_iso("2018-02-22T05:09:11A")
    with pytest.raises(ValueError):
        parse_time_iso("")
    with pytest.raises(TypeError):
        parse_time_iso(None)


@pytest.mark.parametrize("fmt", (
    "%.3i",
    "%Y-%m-%d %H%M%.3S %E",
    "%Y%m%d %H:%M:%.3S %z",
    "%D %.3C %~Z",
    "%~D %~.3C %~Z",
))
def test_parse_roundtrip(fmt):
    t = Time(2020, 1, 28, 12, 30, 45.125, UTC)
    str = format(t, fmt)
    print(f"{fmt} → {str}")
    assert parse_time(fmt, str) == t


