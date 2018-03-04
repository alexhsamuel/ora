.. _time_zones:

Time Zones
==========

`TimeZone` represents a time zone, using data loaded from zoneinfo files in the
`tz database <https://en.wikipedia.org/wiki/Tz_database>`_.  See `this link
<https://data.iana.org/time-zones/theory.html>`_ for information about the tz
database and its limitations.

Ora includes and uses a recent copy of the zoneinfo files, distinct from those
typically installed on UNIX-like systems or those installed with `dateutil` or
`pytz`.


Time zone objects
-----------------

The `TimeZone` type accepts any of the following:

- A time zone name.
- Another `TimeZone` instance.
- A `pytz` time zone object.

For example,

    >>> tz = TimeZone("America/New_York")
    >>> z.name
    'America/New_York'

Ora also lets you give a time zone name almost anywhere a time zone is required.

    >>> format_time_iso(now(), "America/New_York")
    '2018-03-02T07:54:12-05:00'

However, using an explicit time zone object is more efficient.

Ora also provied a `UTC` symbol.

    >>> UTC
    TimeZone('UTC')


Localizing
----------

The principle function of a time zone is to localize a time, _i.e._ to convert a
time to a date and daytime, or vice versa. The `to_local()` and `from_local()`
functions do this, as well as the `@` operator (`__matrix_multiply__`) operator.
See :ref:`localization`.


Display time zone
-----------------

Ora stores a "display time zone", which allows you to choose a default time
zone for formatting.

    >>> get_display_time_zone()
    TimeZone('America/New_York')

The display time zone is initialized based on the `TZ` environment variable, but
you may change it with `set_display_time_zone()`.  You may also use the string
`"display"` to indicate the display time zone in any function that requires a
time zone.

    >>> format_time_iso(now(), "display")
    '2018-03-02T07:59:49-05:00'


System time zone
----------------

Ora also attempts to determine the "system time zone", configured by the 
host systme.  If a system time zone cannot be determined, Ora uses UTC.

Use `get_system_time_zone()` to retrieve this, or specify `"system"` as a time
zone name.  You may not change the system time zone.


Offsets
-------

The `at()` method returns information about a time zone's prescriptions at a
given time.  The returned structure can be unpacked as a sequence.  The offset
is the number of seconds to be added to the UTC time to produce the local time.

    >>> tz.at(now())
    TimeZoneParts(offset=-14400, abbreviation='EDT', is_dst=True)
    >>> offset, abbrev, is_dst = tz.at(now())

To obtain time zone information for a local time (a date and daytime *in that
time zone*) instead, use `at_local()`.

    >>> tz.at_local(2016/Mar/18, MIDNIGHT)
    TimeZoneParts(offset=-14400, abbreviation='EDT', is_dst=True)

