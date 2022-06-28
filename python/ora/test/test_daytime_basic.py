import datetime
import itertools
import pytest

import ora
from   ora import Daytime, DAYTIME_TYPES
from   tools import assert_float_equal

DAYTIME_TYPE_PAIRS = tuple(itertools.product(DAYTIME_TYPES, DAYTIME_TYPES))

#-------------------------------------------------------------------------------

def near(a0, a1):
    return (
           a0 == a1 
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
    assert a.second     == 56.789012
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


def test_from_hms2():
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


def test_from_iso():
    a = Daytime("12:30:45.125")
    assert a.hour       == 12
    assert a.minute     == 30
    assert a.second     == 45.125
    assert a.ssm        == 45045.125
    assert a.valid


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
    assert (60 - p.second) < 2 * Daytime.RESOLUTION


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


def test_comparison0():
    assert     Daytime.MIN     == Daytime.MIN
    assert     Daytime.MAX     != Daytime.MIN
    assert     Daytime.MIN     != Daytime.MAX
    assert     Daytime.MAX     == Daytime.MAX

    assert     Daytime.INVALID == Daytime.INVALID
    assert not Daytime.INVALID != Daytime.INVALID

    assert     Daytime.MISSING == Daytime.MISSING
    assert not Daytime.MISSING != Daytime.MISSING


@pytest.mark.parametrize("Daytime0, Daytime1", DAYTIME_TYPE_PAIRS)
def test_comparison_types(Daytime0, Daytime1):
    assert Daytime0.MIN         == Daytime1.MIN
    assert Daytime0.MISSING     == Daytime1.MISSING
    assert Daytime0.INVALID     == Daytime1.INVALID
    assert Daytime0(12,  0,  0) == Daytime1(12,  0,  0)
    assert Daytime0(23, 59, 59) == Daytime1(23, 59, 59)


def test_comparison1():
    a0 = Daytime( 0,  0, 30)
    a1 = Daytime( 0, 30,  0)
    a2 = Daytime(12,  0,  0)

    assert     a0 == a0
    assert     a1 == a1
    assert     a2 == a2

    assert     Daytime.MIN  <  a0
    assert     Daytime.MIN  <= a0
    assert not Daytime.MIN  == a0
    assert     Daytime.MIN  != a0
    assert not Daytime.MIN  >  a0
    assert not Daytime.MIN  >= a0
    
    assert     a0           <  a1
    assert     a0           <= a1
    assert not a0           == a1
    assert     a0           != a1
    assert not a0           >  a1
    assert not a0           >= a1
    
    assert     a1           <  a2
    assert     a1           <= a2
    assert not a1           == a2
    assert     a1           != a2
    assert not a1           >  a2
    assert not a1           >= a2
    
    assert     a2           <  Daytime.MAX
    assert     a2           <= Daytime.MAX
    assert not a2           == Daytime.MAX
    assert     a2           != Daytime.MAX
    assert not a2           >  Daytime.MAX
    assert not a2           >= Daytime.MAX


def test_comparison2():
    daytimes = (
        Daytime.INVALID, Daytime.MISSING,
        Daytime.MIN, 
        Daytime(0, 0, 1), Daytime(0, 1, 0), Daytime(1, 0, 0),
        Daytime.MAX,
    )
    for i0 in range(len(daytimes)):
        for i1 in range(len(daytimes)):
            a0 = daytimes[i0]
            a1 = daytimes[i1]
            assert (i0 == i1) == (a0 == a1)
            assert (i0 != i1) == (a0 != a1)
            assert (i0 <  i1) == (a0 <  a1)
            assert (i0 <= i1) == (a0 <= a1)
            assert (i0 >  i1) == (a0 >  a1)
            assert (i0 >= i1) == (a0 >= a1)


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
    assert a +   1000 == Daytime( 0,  7, 10)


def test_add2():
    a = Daytime(23, 59, 59.999)
    assert near(a + 0.0001, Daytime(23, 59, 59.9991))
    assert near(a + 0.0005, Daytime(23, 59, 59.9995))
    assert near(a + 0.0010, Daytime( 0,  0,  0.0   ))


def test_subtract0():
    a = Daytime( 0,  0,  0)
    assert a - 0 == a
    assert a - 1 == Daytime(23, 59, 59)


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
    

def test_add_bounds():
    assert Daytime.MIN - Daytime.RESOLUTION == Daytime.MAX
    assert Daytime.MAX + Daytime.RESOLUTION == Daytime.MIN


def test_hash():
    daytimes = (
          [ Daytime.MIDNIGHT + n for n in range(0, 86400, 300) ]
        + [Daytime.INVALID, Daytime.MISSING])
    hashes = frozenset( hash(d) for d in daytimes )
    assert len(hashes) > len(daytimes) // 2


def test_format():
    daytime = Daytime(9, 34, 15.625)
    assert format(daytime, "%H%M")                  == "0934"
    assert format(daytime, "%H::%M::%S")            == "09::34::15"
    assert format(daytime, "%H%M%.5S")              == "093415.62500"
    assert format(daytime, "%1H%_p")                == "9am"

    assert format(Daytime.INVALID, "%H%M")          == "INVA"
    assert format(Daytime.INVALID, "%H::%M::%S")    == "INVALID   "
    assert format(Daytime.MISSING, "%H%M%.5S")      == "MISSING     "
    assert format(Daytime.MISSING, "%1H%_p")        == "MIS"


def test_std():
    time = Daytime(9, 34, 15.6257196).std
    
    assert isinstance(time, datetime.time)
    assert time.hour == 9
    assert time.minute == 34
    assert time.second == 15
    assert time.microsecond == 625720

    with pytest.raises(ValueError):
        Daytime.INVALID.std
    with pytest.raises(ValueError):
        Daytime.MISSING.std


def test_str():
    # Don't test for number of trailing zeros.
    assert str(Daytime(0, 0, 0)) == "00:00:00"
    assert str(Daytime(23, 59, 59.99999)) == "23:59:59.99999"


def test_format_basic():
    daytime = Daytime(9, 34, 15.625)
    assert "it's now {}.".format(daytime) == "it's now " + str(daytime) + "."


def test_format_C():
    daytime = Daytime(9, 34, 15.625)
    assert format(daytime, "%C"  ) == "09:34:15"
    assert format(daytime, "%.0C") == "09:34:15."
    assert format(daytime, "%.3C") == "09:34:15.625"
    assert format(daytime, "%.6C") == "09:34:15.625000"
    

def test_format_iso():
    daytime = Daytime(9, 34, 05.625)
    assert ora.format_daytime_iso(daytime    ) == "09:34:05"
    assert ora.format_daytime_iso(daytime,  0) == "09:34:05."
    assert ora.format_daytime_iso(daytime,  3) == "09:34:05.625"
    assert ora.format_daytime_iso(daytime, 12) == "09:34:05.625000000000"


@pytest.mark.xfail
def test_offset():
    assert Daytime.MIN.offset == 0
    assert 0 < Daytime(12, 0, 0).offset < Daytime.MAX.offset


@pytest.mark.xfail
@pytest.mark.parametrize("Daytime", ora.DAYTIME_TYPES)
def test_from_offset(Daytime):
    for y in [
        Daytime.MIN, 
        Daytime(0, 0, 1), 
        Daytime(12, 0, 0), 
        Daytime.MAX,
        Daytime.INVALID,
    ]:
        assert Daytime.from_offset(y.offset) == y


@pytest.mark.parametrize("Daytime", ora.DAYTIME_TYPES)
def test_from_hmsf(Daytime):
    assert Daytime.from_hmsf(0) == Daytime(0, 0, 0)
    assert Daytime.from_hmsf(123456.789).ssm == pytest.approx(Daytime(12, 34, 56.789).ssm, 0.001)
    assert Daytime.from_hmsf(235959.999).ssm == pytest.approx(Daytime(23, 59, 59.999).ssm, 0.001)
    with pytest.raises(ValueError):
        Daytime.from_hmsf(-0.01)
    with pytest.raises(ValueError):
        Daytime.from_hmsf(240000)


