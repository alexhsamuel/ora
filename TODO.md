# py.hh

- provide a wrapper for each Python C API method typedef
  - wrap other special methods

# Date

- Replace datenum with "proleptic Gregorian ordinal" per Python's
  `datetime.date`?
- Maybe make Month, Day, Ordinal one-indexed?

## PyDate

- API

  - `from_week(week_year, week, weekday)`
  - `from_week()` and `from_ordinal()` should accept single sequences
  - `convert()`
  - format method

- Template `PyDate` on `Date` rather than its traits.
- Consider and test invalid vs. exception date classes.
- `today(tz)` function
- comparison with other date representations
- docstrings
- unit tests

# Infrastructure / tech debt

- make Object be an interface-only type; inherit concrete types from PyObject
- move `py.hh` to plynth and merge with other versions
- include a recent tzlib in the distro

