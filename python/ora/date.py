"""
Gregorian calendar dates.

In a specific location, a calendar date corresponds to a period usually, but not
always, 24 hours long.  A calendar date by itself does not represent any
specific time or interval of time.  A date is is not a subtype of a time, nor
_vice versa_.


# Types

- `Date` - 32-bit date with full range over years 1 to 9999
- `Date16` - 16-bit date with range from 1970 to 2149 Jun 4.


# YMDI

A _YMDI_ is a commonly-used encoding of a date as an eight-decimal digit
integer.  The encoding has the typographic form YYYYMMDD, where YYYY are the
digits of the year, MM of the month, and DD of the day.  Alternately, 

    ymdi = 10000 * year + 100 * month + day

For example, 2004-11-02 is represented by the integer 20041102.  

We recommend you avoid usind YMDI where possible, as nearly any arithmetic
operations on it will produce incorrect results.  Use a date type instead.


# Datenum

In Ora, a _datenum_ is a type-independent represetation of a date.  It is
the number of days since 0001-01-01 (January 1 of the year 1 C.E.).

"""

#-------------------------------------------------------------------------------

from   .ext import Date, Date16

__all__ = (
    "Date", 
    "Date16", 
)

#-------------------------------------------------------------------------------

