.. _time_zones:

Time Zones
==========

`TimeZone` represents a time zone, using data loaded from zoneinfo files in the
`tz database <https://en.wikipedia.org/wiki/Tz_database>`_.  See `this link
<https://data.iana.org/time-zones/theory.html>`_ for information about the tz
database and its limitations.


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

The principle function of a time zone is to localize a time, *i.e.* to convert a
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
host system.  If a system time zone cannot be determined, Ora uses UTC.

Use `get_system_time_zone()` to retrieve this, or specify `"system"` as a time
zone name.  Ora cannot change the system time zone.


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


Time zone data
--------------

Ora uses a method similar to the standard library's `zoneinfo` module to choose
an available zoneinfo database, according to these rules:

1. The first entry of `zoneinfo.TZPATH
   <https://docs.python.org/3/library/zoneinfo.html#zoneinfo.TZPATH>`_ that is
   an absolute path to a directory, if any.  By default, `TZPATH` is initialized
   from the `PYTHONTZPATH` environment variable, if set.  Otherwise, it is
   initialized to a Python distribution-specific list of typical paths at which
   system zoneinfo directories are typically located.  On UNIX-like systems,
   this is most commonly `/usr/share/zoneinfo`.

2. Or, the zoneinfo directory installed with the PyPI `tzdata
   <https://tzdata.readthedocs.io/en/latest/>`_ package, if `tzdata` is
   installed.

3. Or, a copy of the zoneinfo database packaged with Ora itself.

In most Python installations, Ora uses the system-installed zoneinfo by default.
To avoid using the system zoneinfo, set `PYTHONTZPATH` to an empty string.  Ora
will use zoneinfo from the `tzdata` package, if installed, else its own copy.

To instruct Ora to use a specific zoneinfo directory, set the `PYTHONTZPATH`
environment variable to the absolute path, or call `set_zoneinfo_dir()`.  Ora
caches loaded time zone objects; any already loaded will not be reloaded if the
zoneinfo directory is changed by `set_zoneinfo_dir()`.

