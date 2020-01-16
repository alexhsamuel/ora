Limitations
===========

Similar to `datetime`, Ora uses the proleptic_ Gregorian calendar, for years
1-9999 only.  Alternate calendars and B.C.E. dates are not provided.  There is
no support for leap seconds, relativistic effects, or astronomical times.
However, time precision of 1 ns or smaller is supported.

.. _proleptic: https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar


Scope
-----

- Requres C++14 and Python 3.6+.
- Tested on Linux and OSX.  Not currently tested on Windows.
- Support for LP64 architectures (so, x86-64 but not x86) only.

