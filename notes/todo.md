# Work List

1. Full parsing support.
   - remaining date fields
   - date parse unit tests
   - ignore modifiers
   - API?
   - fast parse_date_iso(), etc.
1. Basic string parsing for `convert_to_*()` functions.
1. Add default precision to TimeAPI; use for formatting.
1. Make Time(datetime, tz) work for naive datetime.
1. Intro Python documentation in rst.
1. ReadTheDocs support.
1. Relax numpy setup dependency.
1. Revisit type definitions.
   - Benchmark 2^n vs. 10^n types.
   - Add exact us, ns types.
1. Benchmark tick computations.
1. Fix rounding of ora.now(UsecTime).
1. Remove Time.get_parts().
1. Support UTF-8 in format patterns.
1. Build conda package.
1. Rename `InvalidDateError` -> `BadDateError` _et fils_.
1. Remove superflous `extern`.
1. timezone etc. namespace cleanup
1. Daytime and Time rounding functions.  Maybe like Arrow's `floor()`, `ceil()`?
1. Use `fold` attribute per [PEP-495](https://www.python.org/dev/peps/pep-0495/)
1. Clean up C++ Time and localization functions; document.
1. macOS old tzinfo format.
1. Clean up functions_doc and other old-style docstrings.
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


# C++ API

## Time zone

- Interpret and apply the "future transitions" field in tzinfo files.

# Python API

- Rename `DayInterval` to `DayDuration`.
- Make `first` a keyword-only argument everywhere.
- Add wrappers for formatters.  This would be for efficiency?  Or does this go
  in [fixfmt](http://github.com/alexhsamuel/fixfmt)?

## PyDate

- parsing strings
- shifts by year, month, hour, minute
- "Thursday of the last week of the month"-style function
- docstrings
- unit tests

## PyDaytime

- parsing strings

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



## PyDateDuration

## PyTime

## PyTimeDuration

## PyCalendar

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

- Make a 'zoneinfo dir' abstraction; load time zones from there.

- typename -> class

# Maybe / someday

- Add an (upper) "bound" constant for time, date, daytime counts that is
  distinct from invalid and missing, not itself valid, but compares strictly
  greater than every other valid value.
 
