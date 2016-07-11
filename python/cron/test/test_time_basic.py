import datetime

import pytest

import cron
from   cron import *
import data
from   tools import *

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


def test_sub():
    t = (2016/Jul/11, Daytime(8, 32, 15)) @ UTC
    assert t - t == 0.0
    assert (t + 1) - t == 1.0
    assert xeq((t + 12345.675) - t,  12345.675, 8)
    assert xeq((t - 12345.675) - t, -12345.675, 8)
    assert xeq(t - (t + 12345.675), -12345.675, 8)
    assert xeq(t - (t - 12345.675),  12345.675, 8)


def test_hash():
    times = (
          [ Time.MIN + n for n in range(0, 270000000, 3000000) ]
        + [Time.INVALID, Time.MISSING])
    hashes = frozenset( hash(t) for t in times )
    assert len(hashes) > len(times) // 2


