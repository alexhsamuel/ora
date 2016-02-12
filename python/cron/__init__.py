import enum

import pln.py

from   ._ext import *

__all__ = (
    "Date",
    "DateParts",
    "Month",
    "MonthOfYear",
    "NsecTime",
    "SmallDate",
    "SmallTime",
    "Time",
    "Unix32Time",
    "Unix64Time",
    "Weekday",

    "days_per_month",
    "is_leap_year",
    "ordinals_per_year",

    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",

    "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
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


