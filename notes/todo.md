# Tactical

- Release 0.4.0.a


# Small problems

- xfail tests
- functions:
  - `ora.np.daytime_from_offset()`
  - `ora.np.daytime_from_hms()`
  - `ora.np.daytime_from_ssm()`
  - `ora.np.time_from_offset()`
- ufuncs:
  - matmult for `to_local`, `from_local`
  - `ora.np.get_hms()` and `HMS_DTYPE`
  - `ora.np.get_hour()`
  - `ora.np.get_minute()`
  - `ora.np.get_second()`
  - `ora.np.get_ssm()`
  - `ora.np.to_offset()` (daytime)
  - `ora.np.parse_time()`
  - `ora.np.format_time()`
  - `ora.np.parse_date()`
  - `ora.np.format_date()`
  - `ora.np.parse_daytime()`
  - `ora.np.format_daytime()`
- Inconsistency: `ora.np.get_ymd` returns structured array but `ora.np.to_local` returns two arrays.
- What is the story with "parts" types?
  - date + daytime
  - date + daytime + tz
  - date + daytime + tz parts (parsing?)
  - np?

- fix now() rounding problem... if there is one
  - or more likely, fix rounding in last digit of seconds in Time repr
  - and maybe Daytime should have lower resolution

- [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) tests

# Miscellaneous Work List

1. Rounding functions.
1. matplotlib axis integration
1. Use [ryu](https://github.com/ulfjack/ryu/tree/master/ryu) for `format_seconds()`.
1. Fixfmt-style formatter objects.
1. Date, time, daytime range functions.
1. Nth day of month function (or generator!).
1. Span class.
1. Return namedtuple or similar from Calendar.range.
1. Sloppy time and date parsing.
1. Rename Calendar "range" to "domain".
1. Use an interval class for the calendar domain.
1. Come up with a better syntax for the domain the calendar file.
1. Fix `Time.__qualname__` etc.
1. Remove seconds decimal point in repr of time types with no subsecond resolution.
1. Supress trailing zeros in fractional seconds.
1. Better `Time.__repr__`.
1. Should `format(time)` produce the same as `str(time)`?
1. Make Time(datetime, tz) work for naive datetime.
1. Fix `LocalTime`; see below.
1. Make the Python default Time, Date, Daytime types setable.
1. Accept `datetime.timezone` time zones.
1. Accept `dateutil` time zones.
1. Base Python `Time`, `Date`, `Daytime` classes, as markers only.  Move APIs.
1. Make Python Date like Time.
1. Make Python Daytime like Time.
1. When parsing fractional seconds, work in terms of dayticks.
1. One too many digits of second precision?
1. Fixed-offset time zones, corresponding to `datetime.timezone`.
1. Add default precision to TimeAPI; use for formatting.
1. Use `fold` instead of `first` per [PEP-495](https://www.python.org/dev/peps/pep-0495/)
1. Revisit type definitions.
   - Benchmark 2^n vs. 10^n types.
   - Add exact us, ms types.
1. Convert docstrings to rst.
1. Benchmark tick computations.
1. More parsing support.
   - modifiers: pad, str_case, abbreviate
   - figure out how to parse "%S.%f"; see `test_parse_daytime.py:test_usec()`
   - parse_ex variants
   - C++ API?
1. Remove Time.get_parts().
1. Support UTF-8 in format patterns.
1. Rename `InvalidDateError` -> `BadDateError` _et fils_.
1. Remove superflous `extern`.
1. timezone etc. namespace cleanup
1. Daytime and Time rounding functions.  Maybe like Arrow's `floor()`, `ceil()`?
1. Make `std` a method instead of an attribute?  Accept a time zone?
1. Clean up C++ Time and localization functions; document.
1. Clean up old-style docstrings.
1. Update docs for nex, namespaces, includes.
1. C++ constants for months.
1. Default precision for C++ time, daytime formats (fractional secs?).
   Or make the default daytime format mutable?
1. Adjust the definition of TimeTraits.
1. Check Time128 second precision in C++ and Python.
1. In Python 3.6, use variable annotations.
1. Locale support in formatting.
1. Docstrings for Time methods.
1. Test that the display time zone is thread local.
1. Add DTZ, STZ objects for display, system time zones.
1. Cache `Format` instances used by Python.
1. Add missing strftime format codes: %U, %W.
1. Adjust/clean up C++ predefined format strings.
1. Clean up time zone structs.
1. macOS old tzinfo format.
1. Expose calendar contents as numpy arrays.


# Projects

1. Initial Python documentation
   - formatting
   - parsing
   - philosophy
   - tour

1. Fix rounding in formatting.  I _think_ this is how we should do it: Change
   daytime formatting to use dayticks rather than HMS parts.  This also avoids
   the need to convert to floating point seconds.  In the format_iso_time and
   format_iso_daytime, perform the rounding on the time offset, before splitting
   into datenum and daytick.  (This can work for integrated time formatting
   only, not for componentwise formatting!  Rounding has to apply to all
   components at the same time.)

   - Deal with FE_INVALID in time add/sub.
   - Change `Time` to use _signed_ int64 around UNIX epoch?

1. `print(t @ z)` should show time zone offset.
   - Add time zone offset (and name/abbr?) to `LocalTime`.
   - Sync up C++ LoalDatenumDaytick, LocalTime, TimeParts.
   - `LocalTime` should carry datenum, daytick internally?
   - `LocalTime` should construct Date, Daytime lazily?
   - Add formatting for `LocalTime`.

1. NumPy support
   - date type and ufuncs
     - [x] casts
     - [x] arithmetic
     - [x] comparisons (template)
     - [x] is_valid
     - [ ] weekday
     - [x] to/from ymd date
     - [x] to/from ordinal date
     - [x] to/from week date
     - [x] to/from ymdi
     - [ ] range functions
     - [ ] parsing
     - [ ] formatting
     - [ ] specialized arrfuncs (template)
   - daytime type and ufuncs
     - [x] casts
     - [x] arithmetic
     - [x] comparisons (template)
     - [x] is_valid
     - [ ] to/from hms
     - [ ] parsing
     - [ ] formatting
     - [ ] specialized arrfuncs (template)
   - time type and ufuncs
     - [ ] casts (which?)
     - [x] to_offset
     - [x] from_offset
     - [x] `offset_dtype` attribute
     - [x] arithmetic
     - [x] comparison (template)
     - [x] is_valid
     - [ ] parsing
     - [ ] formatting
     - [ ] specialized arrfuncs (template)
   - localization ufuncs
     - [x] `from_local`
     - [x] `to_local`

1. Investigate 'safe integer' math packages for offset arithmetic.
   - https://github.com/dcleblanc/SafeInt
   - https://github.com/robertramey/safe_numerics
   - https://github.com/RobertLeahy/Safe
   - https://msdn.microsoft.com/en-us/library/windows/desktop/ff521693(v=vs.85).aspx
   - others


# NumPy project

1. Convert `setup.py` numpy dependency to extras?
1. Conda package with/without numpy.
1. Travis builds with/without numpy.
1. Either dtypes need to initialize to INVALID, or underlying types must handle
   any bit pattern, so that `np.empty()` doesn't blow things up.

# C++ API

## Time zone

- Interpret and apply the "future transitions" field in tzinfo files.

# Python API

## PyDate

- shifts by year, month, hour, minute
- "Thursday of the last week of the month"-style function
- docstrings
- unit tests

## PyDaytime

## PyTimeZone

- Create a separate zoneinfo object with a load method.

  ```python
  zoneinfo = Zoneinfo("/usr/share/zoneinfo")
  tz = zoneinfo.load("US/Eastern")
  ```

  but we also want a convenience method,

  ```python
  tz = get_time_zone("US/Eastern")
  ```

- Specify an alternate place to load default time zone.
- Rename `TimeZoneParts` to something better, maybe `TimeZoneOffset`?


## PyTime

## PyCalendar

- unit tests for ~, &, |
- docstrings
- docs
- an actual repository of calendar data?
- ufunc versions of calendar functions


# Infrastructure / tech debt

- clean up namespaces
- make Object be an interface-only type; inherit concrete types from PyObject
- figure out how to point at our zoneinfo dir by default in C++ code

## py.hh

- It doesn't really make sense to have C++ types corresponding to the Python
  types, as instances are checked at runtime anyway.  Get rid of the C++ types
  and make Object a very fat interface?  Or make cast() a no-op?

- Provide wrap() template methods for dynamic methods, in addition to the
  current function/static method wrappers.

# Misc

- Remove `std::experimental::optional` everywhere.

- Put back `from_parts()` overloading in date, time, daytime ctors?

- Investigate why `cal` doesn't agree for older dates

# Maybe / someday

- Add an (upper) "bound" constant for time, date, daytime counts that is
  distinct from invalid and missing, not itself valid, but compares strictly
  greater than every other valid value.
 
