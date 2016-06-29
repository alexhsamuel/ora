import datetime

import pytest

import cron
from   cron import *
import data

#-------------------------------------------------------------------------------

def test_min():
    assert     Time.MIN.valid
    assert not Time.MIN.invalid
    assert not Time.MIN.missing


def test_max():
    assert     Time.MAX.valid
    assert not Time.MAX.invalid
    assert not Time.MAX.missing


def test_comparison():
    assert     Time.MIN     == Time.MIN
    assert     Time.MAX     != Time.MIN
    assert     Time.MIN     != Time.MAX
    assert     Time.MAX     == Time.MAX

    assert     Time.INVALID == Time.INVALID
    assert not Time.INVALID != Time.INVALID

    assert     Time.MISSING == Time.MISSING
    assert not Time.MISSING != Time.MISSING

    assert     Time.INVALID < Time.MISSING < Time.MIN < Time.MAX
    assert     Time.MAX > Time.MIN > Time.MISSING > Time.INVALID


def test_zero():
    t = from_local((0, 0), UTC)
    p = t.get_parts(UTC)
    assert p.date.year              == 1
    assert p.date.month             == Jan
    assert p.date.day               == 1
    assert p.daytime.hour           == 0
    assert p.daytime.minute         == 0
    assert p.daytime.second         == 0
    assert p.time_zone.abbreviation == "UTC"
    assert p.time_zone.offset       == 0
    assert p.time_zone.is_dst       == False

    assert t == from_local((1/Jan/1, MIDNIGHT), UTC)


