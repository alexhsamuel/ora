[![Build Status](https://travis-ci.org/alexhsamuel/ora.svg?branch=master)](https://travis-ci.org/alexhsamuel/ora)

Ora is a standalone time and date implementation for C++ and Python.

Ora is currently alpha software; bugs are plentiful, and APIs are subject to
change.


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

Ora currently has the following limitations.

- Support for the
  ([proleptic](https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar))
  Gregorian calendar only.
- Support for years 1 &ndash; 9999 only; no support for B.C.E. dates.
- No support for leap seconds or relativistic effects.
- Support for C++14 and Python 3.6 only.
- Tested on Linux and OSX.  Not currently tested on Windows.
- Support for LP64 architectures (so, x86-64 but not x86) only.


# Installation

Ora is available:
- On [PyPI](https://pypi.python.org/pypi/ora), as souce and OS/X wheels.
- On [GitHub](https://github.com/alexhsamuel/ora).

See [nodes/developing.md] to build from source.

