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
 
If you use a time zone object, Ora lets you use the `@` operator (the "matrix
multiplication" operator) as a shortcut for this function.

    >>> time_zone = TimeZone("America/Chicago")
    >>> time = (date, daytime) @ time_zone

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

You can also unpack resulting object directly into date and daytime components.

    >>> date, daytime = to_local(time, "America/Chicago")

The '@' operator means `to_local` if the left hand argument is a time.

    >>> date, daytime = time @ "America/Chicago"

The `from_local` and `to_local` functions and the `@` operator will accept
time zone names or objects.  See :ref:`time_zones`.

    >>> date, daytime = time @ UTC
    >>> date, daytime = time @ "display"


