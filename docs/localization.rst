.. _localization:

Localization
============

The central feature of Ora is converting between a time, which is an abstract
specification of a particular physical instant, and one of its local
representations.  A local representation is the date and daytime in a particular
time zone which specifies that time.


Time to local
-------------

The `from_local()` function converts a (date, daytime) pair and time zone to a
time.

    >>> date = Date(1963, 11, 22)
    >>> daytime = Daytime(12, 30, 0)
    >>> time = from_local((date, daytime), "America/Chicago")

The resulting time is a physical instant, independent of time zone.  If you
print it, Ora shows the UTC time, as we have no commonly used time
representation independent of time zone.

    >>> time
    Time(1963, 11, 22, 18, 30, 0.00000000, UTC)
 
You could, of course, construct the `Time` instance directly, rather than with a
date and daytime object.  However, this is just a shortcut for `from_local`.

    >>> time = Time(1963, 11, 22, 12, 30, 0, "America/Chicago")


Local to time
-------------

The `to_local()` function is the inverse: given a time and a time zone, it 
computes the local date and daytime representation.  

    >>> time = Time(1963, 11, 22, 18, 30, 0.00000000, UTC)
    >>> local = to_local(time, "America/Chicago")
    >>> date = local.date
    >>> daytime = local.daytime

You can unpack resulting object directly into date and daytime components.

    >>> date, daytime = to_local(time, "America/Chicago")

The resulting object also provides accessors for the individual compoennts.

    >>> to_local(time, "America/Chicago").hour

The `from_local` and `to_local` functions and the `@` operator will accept
time zone names or objects.  See :ref:`time_zones`.

    >>> date, daytime = time @ UTC
    >>> date, daytime = time @ "display"

`DTZ` is a synonym for the display time zone.

    >>> date, daytime = time @ DTZ


`@` operator
------------

Ora the `@` operator ("matrix multiplication") as a shortcut for `from_local`
and `to_local`.

If the left-hand argument is a time object, `@` is the same as `to_local`.

    >>> date, daytime = time @ time_zone

Since Python operators are implemented on specific types, either the time must
be an Ora time object, or the time zone must be an Ora time zone object.  The
other is converted.

    >>> date, daytime = time @ "America/Chicago"
    >>> date, daytime = "2020-01-19T08:15:00Z" @ UTC

The `@` operator can also serve as `from_local`, with a `(date, daytime)` pair
on the left and the time zone on the right.

    >>> time_zone = TimeZone("America/Chicago")
    >>> time = (date, daytime) @ time_zone

You can use the result of `to_local` on the left, for conversion from date,
daytime in one time zone to date, daytime in another time zone.

    >>> date, daytime = (date, daytime) @ z0 @ z1

The converts a date and daytime in zone `z0` to a (location-independent) time,
then converts this to a date and daytime in zone `z1`.

