import pytest

import ora

#-------------------------------------------------------------------------------

WEEKDAYS = {ora.Mon, ora.Tue, ora.Wed, ora.Thu, ora.Fri}

@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_all_cal(Date):
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
    all_cal = ora.make_const_calendar(date_range, True)

    assert Date.INVALID not in all_cal
    assert Date.MISSING not in all_cal

    for i in range(365):
        assert Date(2018, 1, 1) + i in all_cal

    with pytest.raises(ValueError):
        Date(2017, 12, 31) in all_cal
    with pytest.raises(ValueError):
        Date(2019,  1,  1) in all_cal


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_none_cal(Date):
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
    all_cal = ora.make_const_calendar(date_range, False)

    assert Date.INVALID not in all_cal
    assert Date.MISSING not in all_cal

    for i in range(365):
        assert Date(2018, 1, 1) + i not in all_cal

    with pytest.raises(ValueError):
        Date(2017, 12, 31) in all_cal
    with pytest.raises(ValueError):
        Date(2019,  1,  1) in all_cal


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_weekday_cal(Date):
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
    weekday_cal = ora.make_weekday_calendar(date_range, WEEKDAYS)

    assert Date.INVALID not in weekday_cal
    assert Date.MISSING not in weekday_cal

    for i in range(365):
        date = Date(2018, 1, 1) + i
        assert (date in weekday_cal) == (date.weekday in WEEKDAYS)


@pytest.mark.parametrize("Date", ora.DATE_TYPES)
def test_weekday_cal_range(Date):
    date_range = Date(2018, 1, 1), Date(2018, 12, 31)
    weekday_cal = ora.make_weekday_calendar(date_range, WEEKDAYS)

    assert weekday_cal.range.start == date_range[0]
    assert weekday_cal.range.stop  == date_range[1] + 1  # FIXME: Don't use slice.

    with pytest.raises(ValueError):
        Date(2017, 12, 31) in weekday_cal
    with pytest.raises(ValueError):
        Date(2019,  1,  1) in weekday_cal



