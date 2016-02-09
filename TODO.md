# C++ API

## Date

- Split `DateParts` into {year, month, day, weekday}, `OrdinalDateParts`, and 
  `WeekDateParts`
- Make Date::MAX distinct from Date::INVALID.
- Maybe make Month, Day, Ordinal one-indexed?
- Split week date out of parts?

# Python API

- Stop using setuptools; port make rules from fixfmt.
- Rename `DayInterval` to `DayDuration`.

## PyDate

- Make Date::MAX distinct from Date::INVALID.

- API

  - `__add__()` and other shifts (in days).  Or use `__rshift__()`?
  - Rationalize C++ and Python APIs.
  - `from_week_date()` and `from_ordinal()` should accept single sequences

- Consider and test invalid vs. exception date classes.
- `__format__()` method and support
- `is_leap_year(year)`, `days_in_month(year, month)`, `days_in_year(year)`
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

- make Object be an interface-only type; inherit concrete types from PyObject
- include a recent tzlib in the distro

## py.hh

- move `py.hh` to plynth and merge with other versions

# Misc

- investigate why `cal` doesn't agree for older dates

