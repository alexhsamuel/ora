import datetime

import pytest

import cron
from   cron import *
import data

#-------------------------------------------------------------------------------

def test_utc0():
    t = from_local((1973/Dec/3, 7200.125), UTC)
    # FIXME
    assert(str(t) == "1973-12-03T02:00:00.12500000Z")


def test_nsec_time():
    t = from_local((1973/Dec/3, 7200.125), UTC, Time=NsecTime)
    # FIXME
    assert(str(t) == "1973-12-03T02:00:00.1250000000Z")


    
