import pytest

import ora

#-------------------------------------------------------------------------------

WEEKDAYS = {ora.Mon, ora.Tue, ora.Wed, ora.Thu, ora.Fri}

@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_all_cal(Date):
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
    cal = ora.make_const_calendar(date_range, True)

    for i in range(365):
        assert Date(2018, 1, 1) + i in cal

    with pytest.raises(ValueError):
        Date(2017, 12, 31) in cal
    with pytest.raises(ValueError):
        Date(2019,  1,  1) in cal


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_none_cal(Date):
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
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
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
    cal = ora.make_weekday_calendar(date_range, WEEKDAYS)

    for i in range(365):
        date = Date(2018, 1, 1) + i
        assert (date in cal) == (date.weekday in WEEKDAYS)
        assert cal.contains(date) == (date.weekday in WEEKDAYS)


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_weekday_cal_before_after(Date):
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
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
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
    cal = ora.make_weekday_calendar(date_range, WEEKDAYS)

    assert cal.range.start == date_range[0]
    assert cal.range.stop  == date_range[1] + 1  # FIXME: Don't use slice.

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
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
    cal = ora.make_weekday_calendar(date_range, WEEKDAYS)

    assert cal.shift(20180222, 20) == Date(2018, 3, 22)
    assert cal.shift(20180513, -9) == Date(2018, 5, 1)


