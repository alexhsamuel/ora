# py.hh

- provide a wrapper for each Python C API method typedef
  - wrap other special methods


# Date

- Replace datenum with "proleptic Gregorian ordinal" per Python's
  `datetime.date`?

## PyDate

- API
  - `from_week(week_year, week, weekday)`
  - `from()`
  - `today(tz)` function
- comparison with other date representations
- docstrings
- unit tests
- direct conversion from other date representations without datenum getset

# Infrastructure / tech debt

- move `py.hh` to plynth and merge with other versions
- include a recent tzlib in the distro

