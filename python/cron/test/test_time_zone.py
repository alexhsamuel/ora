import pytest

import cron
from   cron import *

#-------------------------------------------------------------------------------

def test_utc():
    assert UTC.name == "UTC"


def test_name():
    assert TimeZone("US/Eastern").name == "US/Eastern"


def test_us_central():
    tz = TimeZone("US/Central")
    H = 3600
    assert tz.at_local(1960/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1960/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(1969/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1969/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(1970/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1970/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(1971/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1971/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(1980/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(1980/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(2000/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(2000/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)
    assert tz.at_local(2020/Jan/ 1, MIDNIGHT) == (-6 * H, "CST", False)
    assert tz.at_local(2020/Jul/ 1, MIDNIGHT) == (-5 * H, "CDT", True)


