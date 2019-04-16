Ora, a time and date library
============================

Overview
--------

- A concise, ergonomic API centered around physical times and localization.
- Batteries included: time zones, calendars.
- Full NumPy support for times and dates and their functions.
- High performance, as good or better than other implementations.

Ora also provides some advanced and experimental features, such as time and date
types in multiple widths; calendars; and C++ library support.

Ora is not a drop-in replacement for `datetime` or other Python time libraries,
but supports easy interoperability.

Source
------

Ora is `hosted on GitHub <http://github.com/alexhsamuel/ora>`_.  See
the `installation instructions
<https://github.com/alexhsamuel/ora/blob/master/README.md#installation>`_.


Tour
----

    >>> time = now()
    >>> print(time)
    2018-03-01T13:07:25.04988400+00:00

    >>> date, daytime = time @ "America/New_York"
    >>> print(date)
    2018-03-01


.. toctree::
   :maxdepth: 1
   :caption: Contents

   times
   dates
   time-zones
   localization
   formatting
   numpy
   calendars
   background
   limitations


Back matter
-----------

* :ref:`genindex`
* :ref:`search`

