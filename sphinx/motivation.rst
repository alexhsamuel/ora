Motivation
==========

Many Python time libraries already exist, including the standard library
`datetime` module.  Why another?

Ora provides:
- An opinionated, concise API centered around physical times.
- Built-in time zones.
- High performance for all operations.
- Multiple widths and precisions for times and dates.
- Rich C++ interoperability.

Ora is not a drop-in replacement for `datetime` or other Python time libraries,
but supports easy interoperability.


Experiments
-----------

Ora is also a testbed for a number of experiments, related to temporal
programming, API design, and Python extension.

1. Tick-based integer time representations, with heavy use of C++ templates and
   inlining, for very high performance.

1. An API designed around physical times rather than date-time representations,
   with support for dates as a distinct concept.

1. Support for time and date types with multiple widths and precisions.  

1. Techniques for wrapping C++ template types as Python types.  Interoperability
   is challenging, as C++ templates do not provide virtual dispatch.

1. Batteries-included, zero dependency C++ and Python libraries, including
   current time zone data.

1. Support for distinct "invalid" and "missing" values.  A C++ API that provides
   function variants that either raise an exception or return an invalid value
   on error.

1. Full support for both scalar and vector (NumPy) operations. [experimental]

1. Rich calendar support. [planned]

Ora does not meet all these goals!  But it is fun to try.

