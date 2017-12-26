# Motivation

Ora is an implementation of dates and times in the Gregorian calendar.  Many
of these already exist.  Ora differs from them by providing, in a single
package,

- An API and defaults that encourage clear thinking about what dates and times
  really represent.
- A common C++ and Python API.
- Fast nonvirtual, inline-able integer implementations with configurable widths
  and precisions.
- A large feature set, motivated by the needs of financial and business
  calculations. 


# Limitations

Ora currently has the following limitations.

- Support for the
  ([proleptic](https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar))
  Gregorian calendar only.
- Support for years 1 &ndash; 9999 only; no support for B.C.E. dates.
- No support for leap seconds or relativistic effects.
- Support for C++14 and Python 3.5 only.
- Tested on Linux and OSX.  Not currently tested on Windows.
- Support for LP64 architectures (so, x86-64 but not x86) only.

Also, ora is not yet finished.


# Setup

To run tests, build Google's gtest (one time only):

```sh
cd external/gtest
make gtest_main.a
```

To unpack the time zone data (one time only):

```
make zoneinfo
```

# Comparison with C++ `std::chrono`

- chrono splits out the duration from the epoch
- chrono uses a nice `ratio` class.
- chrono mixes `now` into the clock description.
- chrono provides user-defined literals
- chrono doesn't provide time of day or date types
- chrono doesn't provide NaT / invalid (good or bad?)


# Comparison with Python `datetime`

- `datetime.datetime` represents times as year, month, date, hour, minute,
  second, microsecond components.  Ora's `Time` stores integral ticks from a
  fixed offset, which means decomposition into components is more expensive, but
  other operations are cheaper.

- `datetime.datetime` can be "localized", with a time zone, in which case it
  _almost_ specifies a physical instant; or it can be "unlocalized", in which
  case it doesn't specify a physical instant at all.  Ora's `Time` always
  represents a unique physical instant, but can be rendered in any time zone.

