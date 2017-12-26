import pytest

import ora
from   ora import *

#-------------------------------------------------------------------------------

def test_days_in_month():
    for year in (1, 100, 400, 1900, 2000, 2001, 2002, 2003, 2004, 2005, 9999):
        assert days_in_month(year, Jan) == 31
        assert days_in_month(year, Feb) == 29 if is_leap_year(year) else 28
        assert days_in_month(year, Mar) == 31
        assert days_in_month(year, Apr) == 30
        assert days_in_month(year, May) == 31
        assert days_in_month(year, Jun) == 30
        assert days_in_month(year, Jul) == 31
        assert days_in_month(year, Aug) == 31
        assert days_in_month(year, Sep) == 30
        assert days_in_month(year, Oct) == 31
        assert days_in_month(year, Nov) == 30
        assert days_in_month(year, Dec) == 31


def test_is_leap_year():
    assert not is_leap_year(   1)
    assert     is_leap_year(   4)
    assert not is_leap_year( 100)
    assert     is_leap_year( 400)
    assert not is_leap_year(1900)
    assert     is_leap_year(2000)
    assert not is_leap_year(2003)
    assert     is_leap_year(2004)
    assert not is_leap_year(9999)
    assert     is_leap_year(9996)

    with pytest.raises(ValueError):
        is_leap_year(0)
        is_leap_year(10000)

    with pytest.raises(TypeError):
        is_leap_year(1973/Dec/3)
        is_leap_year("1900")
        is_leap_year(None)


def test_days_in_year():
    for year in range(1, 10000):
        assert days_in_year(year) == 366 if is_leap_year(year) else 355


