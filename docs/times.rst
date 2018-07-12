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

You can create a `Time` object from a variety of arguments.  Remember that an
instance represents a physical time, so if you specify a date and daytime
representation, you must specify the time zone as well.  

- Year, month, day, hour, minute, second, time zone of a local time.
- A date, daytime, and time zone.
- Another time instance, or an _aware_ `datetime.datetime` instance.
- An ISO 8601 string, or `"MIN"` or `"MAX"`.

The `std` attribute returns the time represented as closely as possible by a
`datetime.datetime` instance.  The instance's `tzinfo` is always explicitly set
to UTC.

    >>> time.std
    datetime.datetime(2018, 7, 12, 19, 32, 39, 791202, tzinfo=datetime.timezone.utc)


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
values.  For the `Time` class, the resolution is approximately 30 ns.

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


Offsets
-------

Internally, Ora represents a time as the number of "ticks" relative to a fixed
epoch time.  You can access the number of ticks with the `offset` property.

    >>> print(t)
    2018-03-02T12:30:00.00000000+00:00
    >>> t.offset
    2135927186207539200

Each tick is a second or a fraction of a second as given by the `DENOMINATOR`
class attribute, and the offset is stored as an unsigned 64-bit integer.  You
may in fact perform arithmetic directly on these offsets.  For example, to add
sixty seconds,

    >>> print(Time.from_offset(t.offset + 60 * t.DENOMINATOR))
    2018-03-02T12:31:00.00000000+00:00

`RESOLUTION` is simply the reciprocal of `DENOMINATOR`.  The default `Time` type
uses ticks of about 30 ns since 0001-01-01T00:00:00+00:00.


Time types
----------

In addition to `Time`, a number of time types are available, each with
a different range and resolution.

=============== ======== =========== ====================
Type            Size     Resolution  Approx Range (years)
=============== ======== =========== ====================
`SmallTime`      32 bits 1 s         1970-2106
`Unix32Time`     32 bits 1 s         1902-2038
`Unix64Time`     64 bits 1 s         0001-9999
`Time`           64 bits 30 ns       0001-9999
`NsTime`         64 bits 1 ns        1677-2262
`HiTime`         64 bits 233 fs      1970-2106
`Time128`       128 bits 54 zs       0001-9999
=============== ======== =========== ====================

These types differ in the epoch, denominator, and integer type used to store the
offset.  For example, `NsTime` stores a time as signed 64-bit integer
nanoseconds since 1970-01-01 UTC midnight.  This representation is often used in
technical applications, and is also the representation used by NumPy's
"datetime64[ns]" dtype.

Convert back and forth using the types themselves.

    >>> t
    Time(2018, 3, 2, 12, 30, 0.00000000, UTC)
    >>> NsTime(t)
    NsTime(2018, 3, 2, 12, 30, 0.000000000, UTC)

If you try to convert a time that doesn't fit, you'll get an `OverflowError`.

    >>> time = Time(2600, 1, 1, 0, 0, 0, UTC)
    >>> NsTime(time)
    OverflowError: time out of range

Most functions that return a time object accept a `Time` argument, which allows
you to specify which time class you want.

    >>> now(Time=Time128)
    Time128(2018, 3, 2, 12, 49, 21.010432000000000, UTC)


