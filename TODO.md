# C++ API

## Date

- Make Date::MAX distinct from Date::INVALID.
- Maybe make Month, Day, Ordinal one-indexed?

# Python API

- Stop using setuptools; port make rules from fixfmt.
- Rename `DayInterval` to `DayDuration`.

## PyDate

- Make Date::MAX distinct from Date::INVALID.

- API

  - Rationalize C++ and Python APIs.
  - `from_week_date()` and `from_ordinal()` should accept single sequences

- Consider and test invalid vs. exception date classes.
- `__format__()` method and support
- shifts by year, month, hour, minute
- "Thursday of the last week of the month"-style function
- `today(tz)` function
- docstrings
- unit tests

## Formatting

...?

## PyTimeZone

## PyDateDuration

## PyTime

## PyTimeDuration

## PyDaytime

## PyCalendar

# Infrastructure / tech debt

- clean up namespaces
- make Object be an interface-only type; inherit concrete types from PyObject
- figure out how to point at our zoneinfo dir by default in C++ code

## py.hh

- move `py.hh` to plynth and merge with other versions

# Misc

- investigate why `cal` doesn't agree for older dates

# Rejected

- Split `DateParts` into {year, month, day, weekday}, `OrdinalDateParts`, and
  `WeekDateParts`.  This doesn't work so well, because the date parts aren't so
  big or expensive to compute, and separating them makes the formatting code
  substantially more complicated.

