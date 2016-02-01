# py.hh

- move `py.hh` to plynth and merge with other versions

# Date

- Replace datenum with "proleptic Gregorian ordinal" per Python's
  `datetime.date`?
- Maybe make Month, Day, Ordinal one-indexed?

## PyDate

- API

  - `from_week(week_year, week, weekday)`
  - `from_week()` and `from_ordinal()` should accept single sequences

- Consider and test invalid vs. exception date classes.
- `__format__()` method and support
- `today(tz)` function
- docstrings
- unit tests

# Infrastructure / tech debt

- make Object be an interface-only type; inherit concrete types from PyObject
- include a recent tzlib in the distro

