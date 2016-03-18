## Python Time Zones

`TimeZone` represents a time zone, using data loaded from zoneinfo files in the [tz database](https://en.wikipedia.org/wiki/Tz_database).  See [this link](https://github.com/eggert/tz/blob/master/Theory) for information about the tz database and its limitations.  

Cron includes and uses a recent copy of the zoneinfo files, distinct from those typically installed on UNIX-like systems or those installed with `pytz`.


### Creating instances

The `TimeZone` constructor accepts any of the following:

- Another `TimeZone` instance.
- A `pytz` time zone object.
- A time zone name.

Most Cron functions that require a time zone parameter accept any of these as well.

```py
>>> tz = TimeZone("America/New_York")
>>> tz
TimeZone('America/New_York')
>>> tz.name
'America/New_York'
```

### Localizing

The principle function of a time zone is to localize a time, _i.e._ to convert a time to a date and daytime, or vice versa. The `to_local()` and `from_local()` functions do this, as well as the `__matrix_multiply__` ("@") operator.  See [python-local.md](python-local.md).


### Offsets

The `at()` method returns information about a time zone's prescriptions at a given time.  The returned structure can be unpacked as a sequence.  The offset is the number of seconds to be added to the UTC time to produce the local time.

```py
>>> tz.at(now())
TimeZoneParts(offset=-14400, abbreviation='EDT', is_dst=True)
>>> offset, abbrev, is_dst = tz.at(now())
```

To obtain time zone information for a local time (a date and daytime _in that time zone_) instead, use `at_local()`.

```py
>>> tz.at_local(2016/Mar/18, MIDNIGHT)
TimeZoneParts(offset=-14400, abbreviation='EDT', is_dst=True)
```

