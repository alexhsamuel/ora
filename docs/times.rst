Times
=====

`Time` is the default type for times.  An instance represents a *physical
instant* independent of time zone.

    >>> t = now()

The standard representation is as a UTC date and daytime.

    >>> print(t)
    2018-03-02T03:57:10.02192398+00:00

You may specify a time by date and daytime components.  However, you *must*
provide a time zone; without this, the components don't specify a physical
instant.

    >>> t = Time(2018, 3, 2, 23, 7, 15, "America/New_York")
    >>> print(t)
    2018-03-03T04:07:15.00000000+00:00

The sixth argument, seconds, may be a `float`; the previous components must be
integers.  The time zone may be a time zone object or name.

Since a time does not carry a time zone with it, the `repr` is always in terms
of UTC components

    >>> t
    Time(2018, 3, 3, 4, 7, 15.00000000, UTC)


Time conversion
---------------




Special times
-------------

A time class is equipped with special invalid and missing values.

    >>> Time.INVALID
    Time.INVALID                        
    >>> Time.MISSING
    Time.MISSING      

The `valid` property is tre for any time that is not invalid or missing.

    >>> now().valid
    True


Class attributes
----------------

A time class provides `MIN` and `MAX` attributes, giving the earliest and latest
representable times.

    >>> Time.MIN
    Time(1, 1, 1, 0, 0, 0.00000000, UTC)
    >>> Time.MAX
    Time(9999, 12, 31, 23, 59, 59.99999997, UTC)

The `RESOLUTION` attribute is the approximate smallest differnece between time
values.  The `DENOMINATOR` attribute is its reciprocal.  For the `Time` class,
the resolution is approximately 30 ns.

    >>> Time.RESOLUTION
    2.9802322387695312e-08



Arithmetic
----------

Addition with times is always by seconds.

    >>> print(t)
    2018-03-03T04:07:15.00000000+00:00
    >>> print(t + 10)
    2018-03-03T04:07:25.00000000+00:00

The difference between two times is likewise the number of seconds between them.

    >>> Time(2018, 3, 4, 4, 7, 15, UTC) - t
    86400.0


Time types
----------

In addition to `Time`, a number of time types are available, each with
a different storage size, range, and resolution

=============== ======== =========== ====================
Type            Size     Resolution  Approx Range (years)
=============== ======== =========== ====================
`SmallTime`      32 bits 1 s         1970-2016
`Unix32Time`     32 bits 1 s         1902-2038
`Unix64Time`     64 bits 1 s         0001-9999
`Time`           64 bits 30 ns       0001-9999
`NsTime`         64 bits 1 ns        1677-2262
`HiTime`         64 bits 233 fs      1970-2016
`Time128`       128 bits 54 zs       0001-9999
=============== ======== =========== ====================
