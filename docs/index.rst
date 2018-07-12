Ora, a time and date library
============================

Ora is a freestanding time and date implementation for C++ and Python.

Ora is `hosted on GitHub <http://github.com/alexhsamuel/ora>`_.  See
the `installation instructions
<https://github.com/alexhsamuel/ora/blob/master/README.md#installation>`_.


Introduction
------------

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

