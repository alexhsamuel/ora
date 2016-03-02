import datetime

import pytest

import cron
from   cron import *
import data

#-------------------------------------------------------------------------------

def test_utc0():
    t = from_local((1973/Dec/3, 7200.125), UTC)
    # FIXME
    assert(str(t) == "1973-12-03T02:00:00.1250000Z")


def test_nsec_time():
    # FIXME
    t = from_local((1900/Jan/1, 0), UTC, Time=NsecTime)
    assert str(t) == "1900-01-01T00:00:00.000000000Z"

    t = from_local((2444/May/29, Daytime.from_parts(1, 53, 3.999999997)), UTC, Time=NsecTime)
    assert str(t) == "2444-05-29T01:53:03.999999997Z"

    t = from_local((2444/May/29, Daytime.from_parts(1, 53, 5)), UTC, Time=NsecTime)
    assert str(t) == "INVALID"

    t = from_local((1973/Dec/3, 7200.125), UTC, Time=NsecTime)
    assert str(t) == "1973-12-03T02:00:00.125000000Z"


    
