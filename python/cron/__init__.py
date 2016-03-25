import enum

import pln.py

from   .ext import *

__all__ = (
    "Date",
    "Date16",
    "DateParts",
    "Daytime",
    "DT",
    "MIDNIGHT",
    "Month",
    "MonthOfYear",
    "NsecTime",
    "SmallDaytime",
    "SmallTime",
    "Time",
    "TimeZone",
    "Unix32Time",
    "Unix64Time",
    "UTC",
    "Weekday",

    "days_per_month",
    "is_leap_year",
    "from_local",
    "now",
    "ordinals_per_year",
    "to_local",
    "to_local_datenum_daytick",
    "today",

    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",

    "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",

    "UNIX_EPOCH",
    )

#-------------------------------------------------------------------------------

class Weekday(enum.IntEnum):
    """
    A day of the (seven-day) week.
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



# Add the days of the week to the module namespace.
globals().update(Weekday.__members__)


class MonthOfYear:
    """
    A calendar month in a specific year.
    """

    def __init__(self, year, month):
        self.__year = year
        self.__month = month


    def __repr__(self):
        return pln.py.format_ctor(self, self.__year, self.__month)


    def __truediv__(self, day):
        return Date.from_parts(self.__year, self.__month, day)



class Month(enum.IntEnum):

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
# FIXME: This may be ill-advised.

class _DaytimeHourMinute:

    def __init__(self, hour, minute):
        self.hour = hour
        self.minute = minute


    def __truediv__(self, second):
        return Daytime.from_parts(self.hour, self.minute, second)



class _DaytimeHour:

    def __init__(self, hour):
        self.hour = hour


    def __truediv__(self, minute):
        return _DaytimeHourMinute(self.hour, minute)



class _Daytime:

    def __truediv__(self, hour):
        return _DaytimeHour(hour)



DT = _Daytime()

#-------------------------------------------------------------------------------
# FIXME: Move these into C++ and extension code?

UNIX_EPOCH = (1970/Jan/1, MIDNIGHT) @ UTC

#-------------------------------------------------------------------------------

def random_date(Date=Date, min=None, max=None):
    from random import randint

    if min is None:
        min = Date.MIN
    else:
        min = Date.convert(min)
    if max is None:
        max = Date.LAST
    else:
        max = Date.convert(max)
    return Date.from_datenum(randint(min.datenum, max.datenum))


