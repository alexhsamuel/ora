from   pathlib import Path
import pytest

import ora
from   ora import Jun, Jul

#-------------------------------------------------------------------------------

WEEKDAYS = {ora.Mon, ora.Tue, ora.Wed, ora.Thu, ora.Fri}

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


