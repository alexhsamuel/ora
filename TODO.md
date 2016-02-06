# C++ API

## Date

- Replace 1200-03-01 datenum with 0001-01-01 "proleptic Gregorian ordinal" per
  Python's `datetime.date`?
- Maybe make Month, Day, Ordinal one-indexed?
- Split week date out of parts?

# Python API

Rename `DayInterval` to `DayDuration`.

## PyDate

- API

  - Rationalize C++ and Python APIs.
  - `from_week_date()` and `from_ordinal()` should accept single sequences

- Consider and test invalid vs. exception date classes.
- `__format__()` method and support
- `is_leap_year(year)`, `days_in_month(year, month)`, `days_in_year(year)`
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

