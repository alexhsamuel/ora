# Work List

1. Clean up C++ Time and localization functions; document.
1. Update docs for nex, namespaces, includes.
1. C++ constants for months.
1. Default precision for C++ time, daytime formats (fractional secs?).
   Or make the default daytime format mutable?
1. Global display time zone in Python API.
1. Adjust the definition of TimeTraits.
1. Make hms_daytime dtype. Add it to the module.
1. Basic string parsing for `convert_to_*()` functions.
1. Full parsing support.
1. `format()` method and `tp_format` for Date, Daytime, Time.
1. Check Time128 second precision in C++ and Python.
1. Rename `InvalidDateError` -> `BadDateError` _et fils_.
1. Daytime and Time rounding functions.
1. Python namespace reorg.


# C++ API

## Time zone

- Interpret and apply the "future transitions" field in tzinfo files.

# Python API

- Rename `DayInterval` to `DayDuration`.
- Make `first` a keyword-only argument everywhere.

## PyDate

- parsing strings

- API

  - Rationalize C++ and Python APIs.

- `__format__()` method and support
- shifts by year, month, hour, minute
- "Thursday of the last week of the month"-style function
- docstrings
- unit tests

## PyDaytime

- parsing strings

## Formatting

...?

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

- Put back `from_parts()` overloading in date, time, daytime ctors?

- Investigate why `cal` doesn't agree for older dates

- Make a 'zoneinfo dir' abstraction; load time zones from there.

# Maybe / someday

- Add an (upper) "bound" constant for time, date, daytime counts that is
  distinct from invalid and missing, not itself valid, but compares strictly
  greater than every other valid value.
 
