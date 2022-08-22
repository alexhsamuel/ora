Dates
=====

`Date` is the default type for dates.

    >>> d = Date(2016, 3, 15)
    >>> print(d)
    2016-03-15


Date parts
----------

The components of date representations are available as attributes.  These
include the default representation, as well as the ordinal date and week date
representations.

    >>> d.year, d.month, d.day
    (2016, Month.Mar, 15)
    >>> d.year, d.ordinal
    (2016, 75)
    >>> d.week_year, d.week, d.weekday
    (2016, 10, Weekday.Tue)

These components are also accessible in the `ymd` attribute, whose value can be
unpacked to produce the ordinary date components.

    >>> year, month, day = d.ymd

There's also a `ymdi` attribute, which contains the date parts encoded in an
eight-digit decimal integer.

    >>> d.ymdi
    20160315


Date literals
-------------

Months and weekdays are both given as enumerations, respectively `Month` and
`Weekday`.  The enumerals have three-letter abbreviated names.

    >>> Thu
    Weekday.Thu
    >>> Oct
    Month.Oct

The month enumerals also define the `__truediv__` operator to provide this
syntactic trick for writing date literals:

    >>> 2016/Mar/15
    Date(2016, Mar, 15)


Date conversion
---------------

The `Date` constructor makes an effort to convert reasonable date
representations to the date type.  These include,

- Instances of other Ora date classes.
- Python `datetime.date` instances.
- NumPy `datetime64[D]` instances.
- An integer between 10000000 and 99999999  is interpreted as a YMDI date.
- A three-element sequence is interpreted as a (year, month, day) triplet.
- A two-element sequence is interpreted as a (year, ordinal) pair.

For example,

    >>> Date(Date16(2016, Mar, 15))
    Date(2016, Mar, 15)
    >>> Date(datetime.date(2016, 3, 15))
    Date(2016, Mar, 15)
    >>> Date(np.datetime64("2016-03-15", "D"))
    Date(2016, Mar, 15)
    >>> Date(20160315)
    Date(2016, Mar, 15)
    >>> Date((2016, 3, 15))
    Date(2016, Mar, 15)
    >>> Date([2016, 75])
    Date(2016, Mar, 15)

Most Ora functions that take a date parameter will accept any of these.

The `std` property produces the corresponding `datetime.date` instance.

    >>> d.std
    datetime.date(2016, 3, 15)


Special dates
-------------

Each date class provides `MIN` and `MAX` attributes, giving the earliest and
latest representable dates.

    >>> print(Date.MIN, Date.MAX)
    0001-01-01 9999-12-31
    >>> print(Date16.MIN, Date16.MAX)
    1970-01-01 2149-06-04

Each class also provides two special date values, distinct from all other dates.

    >>> Date.INVALID
    Date.INVALID
    >>> Date.MISSING
    Date.MISSING

The `valid` property is true for any date that is not invalid or missing.

    >>> Date(2016, 3, 15).valid
    True
    >>> Date.MISSING.valid
    False


Arithemtic
----------

Adding or subtracting from a date shifts the date forward or backward by entire
days.

    >>> print(d + 10)
    2016-03-25
    >>> print(d - 10)
    2016-03-05

The difference between two dates is the number of days between them.

    >>> d - 2016/Jan/1
    74

Note that this value is one smaller than the date's ordinal, 75, since the
ordinal is one-indexed.

Today
-----

The `today()` function returns the current date (based on the system clock) in a
specified time zone.  *The time zone must be specified*, since at any instant
there are always two parts of the earth that are on different dates.

This code was evaluated at approximately 23:00 New York time.

    >>> today("US/Eastern")
    Date(2016, Mar, 15)
    >>> today(UTC)
    Date(2016, Mar, 16)
 

Other date types
----------------

The `Date16` class is similar to `Date`, but uses a 16-bit integer internally,
and therefore has a narrower range of dates it can represent.

    >>> Date16.MIN
    Date16(1970, Jan, 1)
    >>> Date16.MAX
    Date16(2149, Jun, 4)

Convert back and forth using the types themselves.

    >>> d = Date(1973, 12, 3)
    >>> Date16(d)
    Date16(1973, Dec, 3)

If you try to convert a date that doesn't fit, you'll get an `OverflowError`.

    >>> battle_of_hastings = Date(1066, Oct, 14)
    >>> Date16(battle_of_hastings)
    OverflowError: date not in range

Most functions that return a date object accept a `Date` argument.

    >>> today("America/New_York", Date=Date16)
    Date16(2018, Mar, 1)

