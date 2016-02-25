import pytest

import cron
from   cron import *

#-------------------------------------------------------------------------------

def test_utc():
    assert UTC.name == "UTC"


