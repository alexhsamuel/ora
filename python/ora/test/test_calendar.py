import numpy as np
from   pathlib import Path
import pytest

import ora
from   ora import Date, Jan, Feb, Jun, Jul, Mon, Tue, Fri

#-------------------------------------------------------------------------------

WEEKDAYS = {ora.Mon, ora.Tue, ora.Wed, ora.Thu, ora.Fri}

@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_cal(Date):
    start, stop = 2018/Jan/1, 2018/Feb/1
    cal = ora.Calendar(
        (start, stop),
        (2018/Jan/2, 20180105, "2018-01-13", "2018-01-14")
    )

    for date in ora.Range(start, stop):
        assert (date in cal) == (
            date in {2018/Jan/2, 2018/Jan/5, 2018/Jan/13, 2018/Jan/14}
        )


def test_cal_name():
    range = 2018/Jan/1, 2018/Feb/1
    dates = (2018/Jan/2, 20180105, "2018-01-13", "2018-01-14")
    cal = ora.Calendar(range, dates)
    assert cal.name is None
    assert str(cal) == "calendar"

    cal = ora.Calendar(range, dates, name="test cal")
    assert cal.name == "test cal"
    assert str(cal) == "test cal"



@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_all_cal(Date):
    date_range = Date(2018, 1, 1), Date(2019, 1, 1)
    cal = ora.make_const_calendar(date_range, True)

    for i in range(365):
        assert Date(2018, 1, 1) + i in cal

    with pytest.raises(ValueError):
        Date(2017, 12, 31) in cal
    with pytest.raises(ValueError):
        Date(2019,  1,  1) in cal


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_none_cal(Date):
    date_range = Date(2018, 1, 1), Date(2019, 1, 1)
    cal = ora.make_const_calendar(date_range, False)

    for i in range(365):
        assert Date(2018, 1, 1) + i not in cal

    with pytest.raises(ValueError):
        Date.MISSING in cal
    with pytest.raises(ValueError):
        Date.INVALID in cal
    with pytest.raises(ValueError):
        Date(2017, 12, 31) in cal
    with pytest.raises(ValueError):
        Date(2019,  1,  1) in cal


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_weekday_cal_contains(Date):
    date_range = Date(2018, 1, 1), Date(2019, 1, 3)
    cal = ora.make_weekday_calendar(date_range, WEEKDAYS)

    for i in range(365):
        date = Date(2018, 1, 1) + i
        assert (date in cal) == (date.weekday in WEEKDAYS)
        assert cal.contains(date) == (date.weekday in WEEKDAYS)


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_weekday_cal_before_after(Date):
    date_range = Date(2018, 1, 1), Date(2019, 1, 1)
    cal = ora.make_weekday_calendar(date_range, WEEKDAYS)

    for i in range(363):
        date = Date(2018, 1, 2) + i
        if date in cal:
            assert cal.before(date) == date
            assert cal.after(date)  == date
        else:
            before = cal.before(date)
            assert before < date
            assert before.weekday in WEEKDAYS

            after = cal.after(date)
            assert date < after
            assert after.weekday in WEEKDAYS

    assert cal.before(20180925) == Date(2018, 9, 25)
    assert cal.before(20180922) == Date(2018, 9, 21)

    assert cal.after(20180406) == Date(2018, 4, 6)
    assert cal.after(20180407) == Date(2018, 4, 9)


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_weekday_cal_range(Date):
    date_range = Date(2018, 1, 1), Date(2019, 1, 1)
    cal = ora.make_weekday_calendar(date_range, WEEKDAYS)

    assert cal.range == date_range

    with pytest.raises(ValueError):
        Date.MISSING in cal
    with pytest.raises(ValueError):
        Date.INVALID in cal

    with pytest.raises(ValueError):
        Date(2017, 12, 31) in cal
    with pytest.raises(ValueError):
        Date(2019,  1,  1) in cal

    with pytest.raises(ValueError):
        cal.before(Date(2017, 12, 31))
    with pytest.raises(ValueError):
        cal.after(Date(2019,  1,  1))


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_weekday_cal_shift(Date):
    date_range = Date(2018, 1, 1), Date(2019, 1, 1)
    cal = ora.make_weekday_calendar(date_range, WEEKDAYS)

    assert cal.shift(20180222, 20) == Date(2018, 3, 22)
    assert cal.shift(20180513, -9) == Date(2018, 5, 1)


def test_load_calendar_file():
    cal = ora.load_calendar_file(Path(__file__).parent / "june18.cal")
    assert cal.range == (2018/Jun/1, 2018/Jul/1)

    CAL_DATES = {2018/Jun/1, 2018/Jun/3, 2018/Jun/13, 2018/Jun/25, 2018/Jun/26}
    for i in range(30):
        d = 2018/Jun/1 + i
        assert (d in cal) == (d in CAL_DATES)


def test_load_business_calendar():
    cal = ora.load_business_calendar(Path(__file__).parent / "june18.cal")
    assert cal.range == (2018/Jun/1, 2018/Jul/1)

    assert 2018/Jun/ 1 not in cal  # holiday
    assert 2018/Jun/ 2 not in cal  # Sat
    assert 2018/Jun/ 3 not in cal  # Sun and holiday
    assert 2018/Jun/ 4     in cal
    assert 2018/Jun/ 8     in cal
    assert 2018/Jun/ 9 not in cal  # Sat
    assert 2018/Jun/12     in cal
    assert 2018/Jun/13 not in cal  # holiday
    assert 2018/Jun/14     in cal
    assert 2018/Jun/24 not in cal  # Sun
    assert 2018/Jun/25 not in cal  # holiday
    assert 2018/Jun/26 not in cal  # holiday
    assert 2018/Jun/27     in cal
    assert 2018/Jun/30 not in cal  # Sat


def test_ops():
    cal0 = ora.Calendar(
        (20180101, 20180201),
        [20180102, 20180105, 20180107, 20180125, 20180127]
    )
    cal1 = ora.Calendar(
        cal0.range,
        (20180102, 20180103, 20180108, 20180116, 20180123, 20180125, 20180128)
    )

    not0    = ~cal0
    not1    = ~cal1
    cal_or  = cal0 | cal1
    cal_xor = cal0 ^ cal1
    cal_and = cal0 & cal1

    for date in ora.Range(*cal0.range):
        assert (date in cal0) == (date not in not0)
        assert (date in cal1) == (date not in not1)
        assert (date in cal0 or date in cal1) == (date in cal_or)
        assert ((date in cal0) ^ (date in cal1)) == (date in cal_xor)
        assert (date in cal0 and date in cal1) == (date in cal_and)


def test_get_calendar():
    # Make sure we can load a calendar.
    cal = ora.get_calendar("usa-federal-holidays")
    start, stop = cal.range
    assert start < Date(2018, 1, 1)
    assert stop  > Date(2019, 1, 1)


def test_dates_array_const():
    rng = Date(2018, 7, 1), Date(2018, 8, 1)
    cal = ora.make_const_calendar(rng, False)
    arr = cal.dates_array
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (0, )

    cal = ora.make_const_calendar(rng, True)
    arr = cal.dates_array
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (31, )
    assert list(arr) == [ rng[0] + i for i in range(31) ]


def test_dates_array_holidays():
    cal = ora.get_calendar("usa-federal-holidays")
    arr = cal.dates_array

    dates = [ d for d in ora.Range(*cal.range) if d in cal ]
    assert list(arr) == dates


def test_special():
    cal = ora.get_calendar("all")
    cal = ora.get_calendar("none")
    assert len(cal.dates_array) == 0

    cal = ora.get_calendar("Mon-Tue,Fri")
    for date in cal.dates_array[: 32]:
        assert date.weekday in {Mon, Tue, Fri}



