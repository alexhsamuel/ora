import datetime

import pytest

import cron
from   cron import *

import data
from   util import *

#-------------------------------------------------------------------------------

def test_init0():
    a = Daytime()
    assert not a.valid


def test_init1():
    a0 = Daytime.from_daytick(0)
    a1 = Daytime(a0)
    assert a1.daytick    == 0

    a0 = Daytime.from_daytick(1234567890)
    a1 = Daytime(a0)
    assert a1.daytick    == 1234567890


@pytest.mark.xfail
def test_init_from_time():
    t = datetime.time(0)
    a = Daytime(t)
    assert a.hour       ==  0
    assert a.minute     ==  0
    assert a.second     ==  0
    assert a.valid
    
    t = datetime.time(12, 34, 56, 789012)
    a = Daytime(t)
    assert a.hour       == 12
    assert a.minute     == 34
    assert a.second     == 56.7890123
    assert a.valid
    
    t = datetime.time(23, 59, 59, 999999)
    a = Daytime(t)
    assert a.hour       == 23
    assert a.minute     == 59
    assert a.second     == 59.999999
    assert a.valid
    

def test_from_daytick0():
    a = Daytime.from_daytick(0)
    assert a.hour       ==  0
    assert a.minute     ==  0
    assert a.second     ==  0
    assert a.daytick    ==  0
    assert a.ssm        ==  0
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_from_daytick1():
    # Specific to Daytime.
    tick = 86400 * (1 << 47) - (1 << 44)  # 1/8 sec before midnight
    a = Daytime.from_daytick(tick)
    assert a.daytick    == tick
    assert a.hour       == 23
    assert a.minute     == 59
    assert a.second     == 59.875
    assert a.ssm        == 86399.875
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_from_parts0():
    a = Daytime.from_parts(0, 0, 0)
    assert a.hour       ==  0
    assert a.minute     ==  0
    assert a.second     ==  0
    assert a.daytick    ==  0
    assert a.ssm        ==  0
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_from_parts1():
    a = Daytime.from_parts(12, 34, 56.75)
    assert a.hour       == 12
    assert a.minute     == 34
    assert a.second     == 56.75
    assert a.ssm        == 45296.75
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_from_parts1():
    a = Daytime.from_parts(23, 59, 59.999)
    assert a.hour       == 23
    assert a.minute     == 59
    assert_float_equal(a.second, 59.999)
    assert a.ssm        == 86399.999
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_from_ssm0():
    a = Daytime.from_ssm(0)
    assert a.hour       ==  0
    assert a.minute     ==  0
    assert a.second     ==  0
    assert a.daytick    ==  0
    assert a.ssm        ==  0
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_from_ssm1():
    a = Daytime.from_ssm(1234.5)
    assert a.hour       ==  0
    assert a.minute     == 20
    assert a.second     == 34.5
    assert a.ssm        == 1234.5
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_from_ssm2():
    a = Daytime.from_ssm(86399.875)
    assert a.hour       == 23
    assert a.minute     == 59
    assert a.second     == 59.875
    assert a.ssm        == 86399.875
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_parts0():
    assert Daytime.MIN.parts == (0, 0, 0)

    p = Daytime.from_parts(12, 34, 56.5).parts
    assert p.hour == 12
    assert p.minute == 34
    assert p.second == 56.5
    assert p == (12, 34, 56.5)

    p = Daytime.LAST.parts
    assert p.hour == 23
    assert p.minute == 59
    assert (60 - p.second) < 2 * Daytime.EPSILON


def test_invalid():
    a = Daytime.INVALID
    assert not (0 <= a.hour < 24)
    assert not (0 <= a.minute < 60)
    assert not (0 <= a.second < 60)
    assert not a.valid
    assert a.invalid
    assert not a.missing


def test_missing():
    a = Daytime.MISSING
    assert not (0 <= a.hour < 24)
    assert not (0 <= a.minute < 60)
    assert not (0 <= a.second < 60)
    assert not a.valid
    assert not a.invalid
    assert a.missing


def test_min():
    a = Daytime.MIN
    assert a.hour == 0
    assert a.minute == 0
    assert a.second == 0
    assert a.ssm == 0
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_last():
    a = Daytime.LAST
    assert a.hour == 23
    assert a.minute == 59
    assert (60 - a.second) < 0.000001
    assert (86400 - a.ssm) < 0.000001
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_max():
    a = Daytime.MAX
    assert not a.valid


def test_is_same():
    a = Daytime.from_parts(12, 34, 56.78)
    assert     a.is_same(a)
    assert not a.is_same(Daytime.MISSING)
    assert not a.is_same(Daytime.INVALID)

    assert     Daytime.INVALID.is_same(Daytime.INVALID)
    assert not Daytime.INVALID.is_same(Daytime.MISSING)
    assert not Daytime.INVALID.is_same(a)
    assert not Daytime.MISSING.is_same(Daytime.INVALID)
    assert not Daytime.MISSING.is_same(a)
    assert     Daytime.MISSING.is_same(Daytime.MISSING)


def test_comparison0():
    assert     Daytime.MIN     == Daytime.MIN
    assert     Daytime.LAST    != Daytime.MIN
    assert     Daytime.MIN     != Daytime.LAST
    assert     Daytime.LAST    == Daytime.LAST
    assert     Daytime.MIN.is_same(Daytime.MIN)
    assert     Daytime.LAST.is_same(Daytime.LAST)
    assert     Daytime.MAX.is_same(Daytime.MAX)
    assert not Daytime.MIN.is_same(Daytime.LAST)
    assert not Daytime.LAST.is_same(Daytime.MAX)
    assert not Daytime.MAX.is_same(Daytime.MIN)

    assert     Daytime.INVALID.is_same(Daytime.INVALID)
    assert not Daytime.INVALID == Daytime.INVALID
    assert not Daytime.INVALID != Daytime.INVALID

    assert     Daytime.MISSING.is_same(Daytime.MISSING)
    assert not Daytime.MISSING == Daytime.MISSING
    assert not Daytime.MISSING != Daytime.MISSING


def test_comparison1():
    a0 = Daytime.from_parts( 0,  0, 30)
    a1 = Daytime.from_parts( 0, 30,  0)
    a2 = Daytime.from_parts(12,  0,  0)

    assert     a0.is_same(a0)
    assert     a1.is_same(a1)
    assert     a2.is_same(a2)

    assert     Daytime.MIN  <  a0
    assert     Daytime.MIN  <= a0
    assert not Daytime.MIN  == a0
    assert     Daytime.MIN  != a0
    assert not Daytime.MIN  >  a0
    assert not Daytime.MIN  >= a0
    assert not Daytime.MIN.is_same(a0)
    
    assert     a0           <  a1
    assert     a0           <= a1
    assert not a0           == a1
    assert     a0           != a1
    assert not a0           >  a1
    assert not a0           >= a1
    assert not a0.is_same(a1)
    
    assert     a1           <  a2
    assert     a1           <= a2
    assert not a1           == a2
    assert     a1           != a2
    assert not a1           >  a2
    assert not a1           >= a2
    assert not a1.is_same(a2)
    
    assert     a2           <  Daytime.LAST
    assert     a2           <= Daytime.LAST
    assert not a2           == Daytime.LAST
    assert     a2           != Daytime.LAST
    assert not a2           >  Daytime.LAST
    assert not a2           >= Daytime.LAST
    assert not a2.is_same(Daytime.LAST)


@pytest.mark.xfail
def test_max():
    a2 = Daytime.from_parts(12,  0,  0)
    assert     a2           <  Daytime.MAX
    assert     a2           <= Daytime.MAX
    assert not a2           == Daytime.MAX
    assert     a2           != Daytime.MAX
    assert not a2           >  Daytime.MAX
    assert not a2           >= Daytime.MAX
    assert not a2.is_same(Daytime.MAX)


