import pytest

from   ora import Date, Date16, DateFmt, TimeFmt, TIME_TYPES

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


@pytest.mark.parametrize("Time", TIME_TYPES)
def test_time_fmt_invalid(Time):
    fmt = TimeFmt(-1, invalid="!!!!!!!!", missing="????????????")
    assert fmt(Time.MISSING) == "????????????             "
    assert fmt(Time.INVALID) == "!!!!!!!!                 "

    fmt = TimeFmt(2, invalid="!!!!!!!!", missing="????????????")
    assert fmt(Time.MISSING) == "????????????                "
    assert fmt(Time.INVALID) == "!!!!!!!!                    "


