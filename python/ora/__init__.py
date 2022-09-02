import contextlib
import os
from   pathlib import Path
import re
import warnings

from   .calendar import (
    load_calendar_file, load_business_calendar, CalendarDir,
    format_calendar, dump_calendar_file,
    get_calendar_dir, set_calendar_dir, get_calendar,
)
from   .ext import *
from   .weekday import *
from   .util import Range

__version__ = "0.7.1"

__all__ = (
    "Calendar",

    "Date",
    "Date16",
    "DATE_TYPES",

    "Daytime",
    "Daytime32",
    "UsecDaytime",
    "DAYTIME_TYPES",

    "HiTime",
    "NsTime",
    "SmallTime",
    "Time",
    "Time128",
    "Unix32Time",
    "Unix64Time",
    "TIME_TYPES",

    "HmsDaytime",
    "Month",
    "MonthOfYear",
    "TimeZone",
    "Weekday",
    "YmdDate",

    "days_in_month",
    "days_in_year",
    "format_daytime_iso",
    "format_time",
    "format_time_iso",
    "from_local",
    "get_display_time_zone",
    "get_system_time_zone",
    "get_zoneinfo_dir",
    "is_leap_year",
    "list_zoneinfo_dir",
    "make_const_calendar",
    "make_weekday_calendar",
    "now",
    "parse_calendar",
    "parse_date",
    "parse_daytime",
    "parse_time",
    "parse_time_iso",
    "set_display_time_zone",
    "set_zoneinfo_dir",
    "to_local",
    "to_weekday",
    "today",

    "CalendarDir",
    "dump_calendar_file",
    "format_calendar",
    "get_calendar",
    "get_calendar_dir",
    "load_business_calendar",
    "load_calendar_file",
    "set_calendar_dir",

    "MIDNIGHT",
    "UTC",
    "DTZ",

    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",

    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",

    "display_time_zone",
    "UNIX_EPOCH",
    )

#-------------------------------------------------------------------------------

TIME_TYPES = frozenset((
    HiTime,
    NsTime,
    SmallTime,
    Time,
    Time128,
    Unix32Time,
    Unix64Time,
))

DATE_TYPES = frozenset((
    Date,
    Date16,
))

DAYTIME_TYPES = frozenset((
    Daytime,
    Daytime32,
    UsecDaytime,
))


# Set the location of the time zone database.  If ZONEINFO is set in the
# environment, use it; otherwise, use our own copy of the database.
_INTERNAL_ZONEINFO_DIR = Path(__file__).parent / "zoneinfo"
try:
    set_zoneinfo_dir(os.environ.get("ZONEINFO", str(_INTERNAL_ZONEINFO_DIR)))
except FileNotFoundError as err:
    warnings.warn(
        "missing zoneinfo; check installation or $ZONEINFO: {}".format(err))

#-------------------------------------------------------------------------------

class ParseError(ValueError):
    """
    An error while parsing a date, time, or daytime.
    """

    pass



class MonthOfYear:
    """
    A calendar month in a specific year.

      >>> MonthOfYear(1973, Dec)
      MonthOfYear(1973, Dec)

    Division is overloaded so that division of a year and month produce an
    instance of `MonthOfYear`.

      >>> 1973 / Dec
      MonthOfYear(1973, Dec)

    Division is further overloaded to return a specific day of that month.

      >>> MonthOfYear(1973, Dec) / 3
      Date(1973, Dec, 3)

    Combining these,

      >>> 1973 / Dec / 3
      Date(1973, Dec, 3)

    """

    def __init__(self, year, month):
        self.__year = year
        self.__month = month


    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.__year, self.__month.name)


    def __truediv__(self, day):
        return Date.from_ymd(self.__year, self.__month, day)



class Month(enum.IntEnum):
    """
    A Gregorian month.
    """

    Jan =  1
    Feb =  2
    Mar =  3
    Apr =  4
    May =  5
    Jun =  6
    Jul =  7
    Aug =  8
    Sep =  9
    Oct = 10
    Nov = 11
    Dec = 12

    def __repr__(self):
        return super().__str__()


    def __str__(self):
        return self.name


    def __rtruediv__(self, year):
        return MonthOfYear(year, self)



# Add the months to the module namespace.
globals().update(Month.__members__)


#-------------------------------------------------------------------------------
# FIXME: Move these into C++ and extension code?

MIDNIGHT = Daytime(0, 0, 0)
UNIX_EPOCH = (1970/Jan/1, MIDNIGHT) @ UTC

# Display time zone.
DTZ = "display"

#-------------------------------------------------------------------------------

def random_date(Date=Date, min=None, max=None):
    """
    Returns a random date between `min` and `max`, inclusive.

    @param Date
      The date type to return.
    @param min
      The earliest date to return.  If `None`, uses `Date.MIN`.
    @param max
      The latest date to return.  If `None`, uses `Date.MAX`.
    """
    from random import randint

    if min is None:
        min = Date.MIN
    else:
        min = Date.convert(min)
    if max is None:
        max = Date.MAX
    else:
        max = Date.convert(max)
    return Date.from_datenum(randint(min.datenum, max.datenum))


@contextlib.contextmanager
def display_time_zone(time_zone):
    """
    Context manager to set and restore the display time zone.
    """
    time_zone = TimeZone(time_zone)
    old = get_display_time_zone()
    try:
        set_display_time_zone(time_zone)
        yield
    finally:
        set_display_time_zone(old)


def get_zoneinfo_version():
    """
    Returns the version of the zoneinfo database, if available.

    The version may be stored in a file `+VERSION` in the zoneinfo directory.

    :return:
      The zoneinfo database version, if available, otherwise `None`.
    """
    path = Path(get_zoneinfo_dir()) / "+VERSION"
    try:
        with path.open("rt") as file:
            version = file.readline().strip()
    except FileNotFoundError:
        return None

    if re.match(r"20\d\d[a-z]$", version) is None:
        raise RuntimeError("unexpected zoneinfo version: {}".format(version))
    else:
        return version


def list_zoneinfo_dir(path=None):
    """
    Lists time zones in a zoneinfo directory.

    :param path:
      Zoneinfo directory.  If none, uses `get_zoneinfo_dir()`.
    :return:
      Iterable of time zone names.
    """
    root = Path(get_zoneinfo_dir() if path is None else path)
    for dir, _, names in os.walk(root):
        parts = Path(dir).relative_to(root).parts
        for name in names:
            if "." in name or name in ("leapseconds", "+VERSION"):
                # These are other data files, not zoneinfo entries.
                continue
            yield "/".join((*parts, name))
