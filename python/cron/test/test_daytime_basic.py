import datetime

import pytest

import cron
from   cron import *

import data
from   util import *

#-------------------------------------------------------------------------------

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


