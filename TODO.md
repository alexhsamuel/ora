# py.hh

- provide a wrapper for each Python C API method typedef
  - wrap `tp\_str`, `tp\_repr`
  - wrap other special methods


# Date

- Replace datenum with "proleptic Gregorian ordinal" per Python's
  `datetime.date`?

## PyDate

- `from_ordinal(year, ordinal)`
- `from_week(week_year, week, weekday)`
- `ensure()`
- conversion from other date representations
- comparison with other date representations
- `today(tz)`
- docstrings
- unit tests

# Infrastructure / tech debt

- move `py.hh` to plynth and merge with other versions

