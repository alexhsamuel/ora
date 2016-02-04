# C++ API

## Date

- Replace datenum with "proleptic Gregorian ordinal" per Python's
  `datetime.date`?
- Maybe make Month, Day, Ordinal one-indexed?

# Python API

Rename `DayInterval` to `DayDuration`.
g
## PyDate

- API

  - `from_week_date(week_year, week, weekday)`
  - `from_week_date()` and `from_ordinal()` should accept single sequences

- Consider and test invalid vs. exception date classes.
- `__format__()` method and support
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

