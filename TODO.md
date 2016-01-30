# py.hh

- provide a wrapper for each Python C API method typedef
  - wrap other special methods


# Date

- Replace datenum with "proleptic Gregorian ordinal" per Python's
  `datetime.date`?

## PyDate

- API
  - `from_ordinal(year, ordinal)`
  - `from_week(week_year, week, weekday)`
  - `ensure()`
  - `today(tz)`
- conversion from other date representations
- comparison with other date representations
- docstrings
- unit tests

# Infrastructure / tech debt

- move `py.hh` to plynth and merge with other versions

