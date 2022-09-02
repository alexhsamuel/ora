import numpy as np
import pytest

from   ora import Date, Date16, Time128, DateFmt, TimeFmt, TIME_TYPES

#-------------------------------------------------------------------------------

def test_date_fmt():
    fmt = DateFmt()
    assert fmt((1, 1, 1)) == "0001-01-01"
    assert fmt("2022-08-22") == "2022-08-22"
    assert fmt(Date("1969-12-31")) == "1969-12-31"
    assert fmt(Date16("1970-01-01")) == "1970-01-01"
    assert fmt(Date.INVALID) == "INVALID   "
    assert fmt(Date16.MISSING) == "MISSING   "


def test_date_fmt_invalid():
    fmt = DateFmt(invalid="!!!!!!!!", missing="????????????")
    assert fmt(Date.MISSING) == "??????????"
    assert fmt(Date.INVALID) == "!!!!!!!!  "


def test_time_fmt():
    fmt = TimeFmt(-1)
    assert fmt("2022-08-23T00:07:52.123Z") == "2022-08-23T00:07:52+00:00"
    assert fmt("2022-08-23T00:07:52.789Z") == "2022-08-23T00:07:53+00:00"

    fmt = TimeFmt(1)
    assert fmt("2022-08-23T00:07:52.123Z") == "2022-08-23T00:07:52.1+00:00"
    assert fmt("2022-08-23T00:07:52.789Z") == "2022-08-23T00:07:52.8+00:00"

    fmt = TimeFmt(4)
    assert fmt("2022-08-23T00:07:52.123Z") == "2022-08-23T00:07:52.1230+00:00"
    assert fmt("2022-08-23T00:07:52.789Z") == "2022-08-23T00:07:52.7890+00:00"

    fmt = TimeFmt(8)
    assert fmt("2022-08-23T00:07:52.123Z") == "2022-08-23T00:07:52.12300000+00:00"
    assert fmt("2022-08-23T00:07:52.789Z") == "2022-08-23T00:07:52.78900000+00:00"

    fmt = TimeFmt(9)
    assert fmt("2022-08-23T00:07:52.123Z") == "2022-08-23T00:07:52.123000000+00:00"
    assert fmt("2022-08-23T00:07:52.789Z") == "2022-08-23T00:07:52.789000000+00:00"
    assert fmt("2022-08-23T00:07:52.123456789Z") == "2022-08-23T00:07:52.123456789+00:00"


def test_time_fmt_datetime64():
    t = np.datetime64("2022-09-01T01:23:45.345678901", "ns")
    assert TimeFmt(-1)(t) == "2022-09-01T01:23:45+00:00"
    assert TimeFmt( 0)(t) == "2022-09-01T01:23:45.+00:00"
    assert TimeFmt( 1)(t) == "2022-09-01T01:23:45.3+00:00"
    assert TimeFmt( 2)(t) == "2022-09-01T01:23:45.35+00:00"
    assert TimeFmt( 3)(t) == "2022-09-01T01:23:45.346+00:00"
    assert TimeFmt( 4)(t) == "2022-09-01T01:23:45.3457+00:00"
    assert TimeFmt( 5)(t) == "2022-09-01T01:23:45.34568+00:00"
    assert TimeFmt( 6)(t) == "2022-09-01T01:23:45.345679+00:00"
    assert TimeFmt( 7)(t) == "2022-09-01T01:23:45.3456789+00:00"
    assert TimeFmt( 8)(t) == "2022-09-01T01:23:45.34567890+00:00"
    assert TimeFmt( 9)(t) == "2022-09-01T01:23:45.345678901+00:00"
    assert TimeFmt(10)(t) == "2022-09-01T01:23:45.3456789010+00:00"


def test_time_fmt_128():
    t = Time128("2022-09-01T12:34:56.789012345678+00:00")
    assert TimeFmt(-1)(t) == "2022-09-01T12:34:57+00:00"
    assert TimeFmt( 0)(t) == "2022-09-01T12:34:57.+00:00"
    assert TimeFmt( 1)(t) == "2022-09-01T12:34:56.8+00:00"
    assert TimeFmt( 2)(t) == "2022-09-01T12:34:56.79+00:00"
    assert TimeFmt( 3)(t) == "2022-09-01T12:34:56.789+00:00"
    assert TimeFmt( 4)(t) == "2022-09-01T12:34:56.7890+00:00"
    assert TimeFmt( 5)(t) == "2022-09-01T12:34:56.78901+00:00"
    assert TimeFmt( 6)(t) == "2022-09-01T12:34:56.789012+00:00"
    assert TimeFmt( 7)(t) == "2022-09-01T12:34:56.7890123+00:00"
    assert TimeFmt( 8)(t) == "2022-09-01T12:34:56.78901235+00:00"
    assert TimeFmt( 9)(t) == "2022-09-01T12:34:56.789012346+00:00"


# Currently doesn't work because we use NsTime in the implementation.
@pytest.mark.xfail
def test_time_fmt_128hi():
    t = Time128("2022-09-01T12:34:56.789012345678+00:00")
    assert TimeFmt(10)(t) == "2022-09-01T12:34:56.7890123457+00:00"


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_time_fmt_invalid(Time):
    fmt = TimeFmt(-1, invalid="!!!!!!!!", missing="????????????")
    assert fmt(Time.MISSING) == "????????????             "
    assert fmt(Time.INVALID) == "!!!!!!!!                 "

    fmt = TimeFmt(2, invalid="!!!!!!!!", missing="????????????")
    assert fmt(Time.MISSING) == "????????????                "
    assert fmt(Time.INVALID) == "!!!!!!!!                    "


