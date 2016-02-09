import datetime

import pytest

import cron
from   cron import *

import data

#-------------------------------------------------------------------------------

def test_date_add():
    d = 1973/Dec/3
    assert(d + -720595).invalid
    assert d + -720594 ==    1/Jan/ 1
    assert d + -100000 == 1700/Feb/17
    assert d +  -10000 == 1946/Jul/18
    assert d +   -1000 == 1971/Mar/ 9
    assert d +    -100 == 1973/Aug/25
    assert d +     -10 == 1973/Nov/23
    assert d +      -1 == 1973/Dec/ 2
    assert d +       0 == 1973/Dec/ 3
    assert d +       1 == 1973/Dec/ 4
    assert d +       2 == 1973/Dec/ 5
    assert d +       7 == 1973/Dec/10
    assert d +      10 == 1973/Dec/13
    assert d +      28 == 1973/Dec/31
    assert d +      29 == 1974/Jan/ 1
    assert d +     100 == 1974/Mar/13
    assert d +    1000 == 1976/Aug/29
    assert d +   10000 == 2001/Apr/20
    assert d +  100000 == 2247/Sep/18
    assert d + 1000000 == 4711/Oct/31
    assert d + 2931464 == 9999/Dec/31
    assert(d + 2931465).invalid

    assert(-720595 + d).invalid
    assert -720594 + d ==    1/Jan/ 1
    assert -100000 + d == 1700/Feb/17
    assert      -1 + d == 1973/Dec/ 2
    assert       0 + d == 1973/Dec/ 3
    assert       1 + d == 1973/Dec/ 4
    assert 1000000 + d == 4711/Oct/31
    assert 2931464 + d == 9999/Dec/31
    assert(2931465 + d).invalid

    assert (d + 100) + -100 == d
    assert -100 + (d + 100) == d
    assert 100 + (-100 + d) == d


