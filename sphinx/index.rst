.. ora documentation master file, created by
   sphinx-quickstart on Sun Jan 21 10:10:32 2018.

Ora, a date and time library
============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   times
   dates
   time zones, daytimes, and localization


Motivation
----------

Ora is an implementation of dates and times in the Gregorian calendar.  Many
of these already exist.  Why another?

- A terse and opinionated API that encourages best practice.

- A large feature set with a consistent API, and no additional dependencies.

- Fast C++ nonvirtual, inline integer implementations of times and dates, with
  high-performance hand-written Python wrappers.

Ora can convert to and from dates and times represented with the Python standard
`datetime` library, but is not an extension, nor is it drop-in compatible.



Tour
----

::

    >>> from ora import *
    >>> time = now()
    >>> print(time)
    2018-01-21T15:52:28.19023199Z

    >>> tz = TimeZone("America/New_York")
    >>> date, daytime = time @ tz
    >>> print(date)
    2018-01-21



Limitations
-----------

Ora currently has the following limitations.

- Support for the proleptic_ Gregorian calendar only.
- Support for years 1–9999 only; no support for B.C.E. dates.
- No support for leap seconds.
- No support for astromical or other technical time systems.
- Requres C++14 and Python 3.6+.
- Tested on Linux and OSX.  Not currently tested on Windows.
- Support for LP64 architectures (so, x86-64 but not x86) only.

.. _proleptic: https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar





Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

