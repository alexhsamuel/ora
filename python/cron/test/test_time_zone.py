import pytest

import cron
from   cron import *

#-------------------------------------------------------------------------------

def test_utc():
    assert UTC.name == "UTC"


def test_name():
    assert TimeZone.get("US/Eastern").name == "US/Eastern"



