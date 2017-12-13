# Work List

1. Methods to return `datetime` instances.
1. Rename.  ("ora"?)
1. Synchronize time format to Python datetime.
1. Change formatting.  Instead of %k etc, support:
   - synchronize time format to Python datetime
   - missing C89 directives: %G, %u, %V
   - adjust ISO/RFC predefined format strings
   - add predefines format strings with subsecond accuracy
   - support UTF-8 patterns
1. Add an explicit format() method that takes time zone, localization.
1. Zoneinfo dir:
   - Provide a way to set it programatically.
   - Provide an accessor.
   - Default to our own, for macOS for now.
1. Revisit type definitions.
   - Benchmark 2**n vs. 10**n types.
   - Add exact us, ns types.
1. Cache `Format` instances uses by Python.
1. Fast-track RFC 3339 formatting function.
1. Review built-in types.  
   - Add exact us, ns types. 
   - Benchmark them.
1. Make Time(datetime, tz) work for naive datetime.
1. Make it `pip install`able.
1. Build conda package.
1. Rename `InvalidDateError` -> `BadDateError` _et fils_.
1. Rename aslib namespace.
1. timezone etc. namespace cleanup
1. Basic string parsing for `convert_to_*()` functions.
1. Full parsing support.
1. Change LocalTime to a proper type, with 'year', 'month' etc. passthrough properties
1. Daytime and Time rounding functions.  Maybe like Arrow's `floor()`, `ceil()`?
1. Use `fold` attribute per [PEP-495](https://www.python.org/dev/peps/pep-0495/)
1. Clean up C++ Time and localization functions; document.
1. macOS old tzinfo format.
1. Benchmark.
1. Basic string parsing for `convert_to_*()` functions.
1. Full parsing support.
1. Update docs for nex, namespaces, includes.
1. C++ constants for months.
1. Default precision for C++ time, daytime formats (fractional secs?).
   Or make the default daytime format mutable?
1. Global display time zone in Python API.
1. Adjust the definition of TimeTraits.
1. Check Time128 second precision in C++ and Python.
1. In Python 3.6, use variable annotations.
1. Locale support in formatting.


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
- System and display time zone abstractions.



## PyDateDuration

## PyTime

## PyTimeDuration

## PyCalendar

# Infrastructure / tech debt

- remove `tp_print` hack from `PyDate` and replace with a type registration
  scheme 
- clean up namespaces
- make Object be an interface-only type; inherit concrete types from PyObject
- figure out how to point at our zoneinfo dir by default in C++ code

## py.hh

- move `py.hh` to plynth and merge with other versions

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
 
