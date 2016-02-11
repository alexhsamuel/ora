import pytest

import cron
from   cron import *

#-------------------------------------------------------------------------------

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


