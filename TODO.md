# py.hh

- provide a wrapper for each Python C API method typedef
  - wrap other special methods


# Date

- Replace datenum with "proleptic Gregorian ordinal" per Python's
  `datetime.date`?

## PyDate

- API

  - `from_week(week_year, week, weekday)`
  - `from_week()` and `from_ordinal()` should accept single sequences
  - `from()`
  - format method

- `today(tz)` function
- comparison with other date representations
- docstrings
- unit tests

# Infrastructure / tech debt

- move `py.hh` to plynth and merge with other versions
- include a recent tzlib in the distro

