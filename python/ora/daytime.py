"""
Daytimes = Times of day.

A daytime is a time of day.  Daytime is a representation of a time in a
contextual day and location, as one might read off an ordinary clock.  A daytime
is an approximation to a specific time of day, in the sense that a floating
point number is an approximation to a real number.

Daytimes are not aware of daylight savings time transitions or any other
time zone effects, as neither a date nor a time zone are specified.  A daytime
simply counts forward from midnight for 24 hours.


# Types

- `Daytime` - high-precision 64-bit daytime
- `Daytime32` - 32-bit daytime 


# SSM

_Seconds since midnight_ (SSM) represents a daytime as a `float` number of
seconds since midnight (in the same time zone offset).


# Dayticks

_Dayticks_ represents a daytime as an integral number of ticks, each 2**-47
seconds, since midnight (in the same time zone offset).

"""

#-------------------------------------------------------------------------------

from   .ext import Daytime, Daytime32

__all__ = (
    "Daytime",
    "Daytime32",
)

#-------------------------------------------------------------------------------

