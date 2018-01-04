import contextlib
import enum
import os
from   pathlib import Path
import re
import sys
import warnings

from   .ext import *

__version__ = "0.2.4"

__all__ = (
    "Date",
    "Date16",
    "Daytime",
    "Daytime32",
    "HmsDaytime",
    "MIDNIGHT",
    "Month",
    "MonthOfYear",
    "NsDaytime",
    "NsTime",
    "NsecTime",
    "SmallTime",
    "Time",
    "Time128",
    "TimeZone",
    "UTC",
    "Unix32Time",
    "Unix64Time",
    "Weekday",
    "YmdDate",

    "days_in_month",
    "days_in_year",
    "format_time_iso",
    "from_local",
    "get_display_time_zone",
    "get_system_time_zone",
    "get_zoneinfo_dir",
    "is_leap_year",
    "now",
    "parse_date",
    "parse_daytime",
    "parse_time",
    "parse_time_iso",
    "set_display_time_zone",
    "set_zoneinfo_dir",
    "to_local",
    "to_weekday",
    "today",

    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",

    "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",

    "display_time_zone",
    "UNIX_EPOCH",
    )

#-------------------------------------------------------------------------------

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



#-------------------------------------------------------------------------------

class Weekday(enum.IntEnum):
    """
    A day of the (seven-day) week.

    Integer values are counted from Monday = 0.
    """

    Mon = 0
    Tue = 1
    Wed = 2
    Thu = 3
    Fri = 4
    Sat = 5
    Sun = 6

    def __repr__(self):
        return super().__str__()


    def __str__(self):
        return self.name



# FIXME: Is this a good idea?
def to_weekday(obj):
    if isinstance(obj, Weekday):
        return obj
    try:
        return Weekday[obj]
    except KeyError:
        pass
    try:
        return Weekday(obj)
    except ValueError:
        pass
    raise ValueError("can't convert to a weekday: {!r}".format(obj))


# Add the days of the week to the module namespace.
globals().update(Weekday.__members__)


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

UNIX_EPOCH = (1970/Jan/1, MIDNIGHT) @ UTC

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

    if re.match("20\d\d[a-z]$", version) is None:
        raise RuntimeError("unexpected zoneinfo version: {}".format(version))
    else:
        return version



