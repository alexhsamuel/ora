import datetime

import pytest

import cron
from   cron import *

import data
from   util import *

#-------------------------------------------------------------------------------

def near(a0, a1):
    return (
           a0.is_same(a1)
        or (a0.valid and a1.valid and abs(a0.ssm - a1.ssm) % 86400 < 1e-6)
    )

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


def test_from_hms0():
    a = Daytime.from_hms(0, 0, 0)
    assert a.hour       ==  0
    assert a.minute     ==  0
    assert a.second     ==  0
    assert a.daytick    ==  0
    assert a.ssm        ==  0
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_from_hms1():
    a = Daytime.from_hms(12, 34, 56.75)
    assert a.hour       == 12
    assert a.minute     == 34
    assert a.second     == 56.75
    assert a.ssm        == 45296.75
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_from_hms1():
    a = Daytime.from_hms(23, 59, 59.999)
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
    assert Daytime.MIN.hms == (0, 0, 0)

    p = Daytime(12, 34, 56.5).hms
    assert p.hour == 12
    assert p.minute == 34
    assert p.second == 56.5
    assert p == (12, 34, 56.5)

    p = Daytime.MAX.hms
    assert p.hour == 23
    assert p.minute == 59
    assert (60 - p.second) < 2 * Daytime.EPSILON


def test_invalid():
    a = Daytime.INVALID
    with pytest.raises(ValueError):
        a.hour
    with pytest.raises(ValueError):
        a.minute
    with pytest.raises(ValueError):
        a.second
    assert not a.valid
    assert a.invalid
    assert not a.missing


def test_missing():
    a = Daytime.MISSING
    with pytest.raises(ValueError):
        a.hour
    with pytest.raises(ValueError):
        a.minute
    with pytest.raises(ValueError):
        a.second
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


def test_max():
    a = Daytime.MAX
    assert a.hour == 23
    assert a.minute == 59
    assert (60 - a.second) < 0.000001
    assert (86400 - a.ssm) < 0.000001
    assert a.valid
    assert not a.invalid
    assert not a.missing


def test_is_same():
    a = Daytime(12, 34, 56.78)
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
    assert     Daytime.MAX     != Daytime.MIN
    assert     Daytime.MIN     != Daytime.MAX
    assert     Daytime.MAX     == Daytime.MAX
    assert     Daytime.MIN.is_same(Daytime.MIN)
    assert     Daytime.MAX.is_same(Daytime.MAX)
    assert not Daytime.MIN.is_same(Daytime.MAX)

    assert     Daytime.INVALID.is_same(Daytime.INVALID)
    assert not Daytime.INVALID == Daytime.INVALID
    assert not Daytime.INVALID != Daytime.INVALID

    assert     Daytime.MISSING.is_same(Daytime.MISSING)
    assert not Daytime.MISSING == Daytime.MISSING
    assert not Daytime.MISSING != Daytime.MISSING


def test_comparison1():
    a0 = Daytime( 0,  0, 30)
    a1 = Daytime( 0, 30,  0)
    a2 = Daytime(12,  0,  0)

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
    
    assert     a2           <  Daytime.MAX
    assert     a2           <= Daytime.MAX
    assert not a2           == Daytime.MAX
    assert     a2           != Daytime.MAX
    assert not a2           >  Daytime.MAX
    assert not a2           >= Daytime.MAX
    assert not a2.is_same(Daytime.MAX)


def test_add0():
    a = Daytime( 0,  0,  0)
    assert a +      1 == Daytime( 0,  0,  1)
    assert a +     10 == Daytime( 0,  0, 10)
    assert a +    100 == Daytime( 0,  1, 40)
    assert a +   1000 == Daytime( 0, 16, 40)
    assert a +  10000 == Daytime( 2, 46, 40)
    assert a + 100000 == Daytime( 3, 46, 40)


def test_add1():
    a = Daytime(23, 50, 30)
    assert a +      1 == Daytime(23, 50, 31)
    assert a +    100 == Daytime(23, 52, 10)
    assert a +  10000 == Daytime( 2, 37, 10)


def test_add2():
    a = Daytime(23, 59, 59.999)
    assert near(a + 0.0001, Daytime(23, 59, 59.9991))
    assert near(a + 0.0005, Daytime(23, 59, 59.9995))
    assert near(a + 0.0010, Daytime( 0,  0,  0.0000))
    assert near(a + 0.0100, Daytime( 0,  0,  0.0090))
    assert near(a + 1     , Daytime( 0,  0,  0.9990))


def test_subtract0():
    a = Daytime( 0,  0,  0)
    assert a -      1 == Daytime(23, 59, 59)
    assert a -     10 == Daytime(23, 59, 50)
    assert a -    100 == Daytime(23, 58, 20)
    assert a -   1000 == Daytime(23, 43, 20)
    assert a -  10000 == Daytime(21, 13, 20)
    assert a - 100000 == Daytime(20, 13, 20)


def test_subtract1():
    a = Daytime(23, 50, 30)
    assert a -      1 == Daytime(23, 50, 29)
    assert a -    100 == Daytime(23, 48, 50)
    assert a -  10000 == Daytime(21,  3, 50)


def test_subtract2():
    a = Daytime(23, 59, 59.999)
    assert near(a - 0.0001, Daytime(23, 59, 59.9989))
    assert near(a - 0.0005, Daytime(23, 59, 59.9985))
    assert near(a - 0.0010, Daytime(23, 59, 59.9980))
    assert near(a - 0.0100, Daytime(23, 59, 59.9890))
    assert near(a - 1     , Daytime(23, 59, 58.9990))
    

