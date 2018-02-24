[![Build Status](https://travis-ci.org/alexhsamuel/ora.svg?branch=master)](https://travis-ci.org/alexhsamuel/ora)

Ora is a standalone time and date implementation for C++ and Python.


# Motivation

Many Python time libraries already exist, including the standard library
`datetime` module.  

Ora provides:
- An opinionated, concise API centered around physical times.
- Built-in time zones.
- High performance for all operations.
- Multiple widths and precisions for times and dates.
- Rich C++ interoperability.

Ora is not a drop-in replacement for `datetime` or other Python time libraries,
but supports easy interoperability.


# Limitations

Ora is currently alpha software; bugs are plentiful, and APIs are subject to
change.

### Scope

Similar to `datetime`, Ora uses the
([proleptic](https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar))
Gregorian calendar, for years 1 &ndash; 9999 only.  Alternate calendars and
B.C.E. dates are not provided.  There is no support for leap seconds,
relativistic effects, or astronomical times.

### Platform

- Requires C++14 and Python 3.6+.
- Tested on Linux and OSX.  Not currently tested on Windows.
- Tested on x86-64 only.


# Installation

Ora is available:
- On [PyPI](https://pypi.python.org/pypi/ora), as souce and OS/X wheels.
- On [GitHub](https://github.com/alexhsamuel/ora).

See [developing.md](docs/developing.md) to build from source.

