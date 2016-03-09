# C++ API

## Date

- Maybe make Month, Day, Ordinal one-indexed?

## Time zone

- Rethink memory management for loaded `TimeZone` objects

# Python API

- Stop using setuptools; port make rules from fixfmt.
- Rename `DayInterval` to `DayDuration`.

## PyDate

- parsing strings

- API

  - Rationalize C++ and Python APIs.
  - `from_week_date()` and `from_ordinal()` should accept single sequences
  - `__repr__()` should return something reasonable

- Consider and test invalid vs. exception date classes.
- `__format__()` method and support
- shifts by year, month, hour, minute
- "Thursday of the last week of the month"-style function
- `today(tz)` function
- docstrings
- unit tests

## PyDaytime

- Core type:

    - conversion from `datetime.time`

- parsing strings

## Formatting

...?

## PyTimeZone

Create a separate zoneinfo object with a load method.

```python
zoneinfo = Zoneinfo("/usr/share/zoneinfo")
tz = zoneinfo.load("US/Eastern")
```

but we also want a convenience method,

```python
tz = get_time_zone("US/Eastern")
```

While we're at it, we probably need to start using `std::shared_ptr` for time
zone objects.

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

- Use some type registration system for more efficient conversion from one
  Python type template to another, e.g. in `convert_date_object()` (to replace
  the "tp_print hack").

- Make a 'zoneinfo dir' abstraction; load time zones from there.

# Maybe / someday

- Add an (upper) "bound" constant for time, date, daytime counts that is
  distinct from invalid and missing, not itself valid, but compares strictly
  greater than every other valid value.
 

# Rejected

- Split `DateParts` into {year, month, day, weekday}, `OrdinalDateParts`, and
  `WeekDateParts`.  This doesn't work so well, because the date parts aren't so
  big or expensive to compute, and separating them makes the formatting code
  substantially more complicated.

