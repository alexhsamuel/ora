NumPy
=====

Each Ora time, date, and daytime type is also a NumPy scalar, and carries a
corresponding dtype.  Many Ora functions are also available as ufuncs or
vectorized functions, which operate efficiently over NumPy arrays.


Arrays
------

NumPy will choose the Ora dtype automatically for an array of the corresponding
type.

    >>> arr = np.array([now(), now(), now()])
    >>> arr.dtype
    dtype(Time)

You can also get the dtype from the `dtype` attribute.

    >>> Time.dtype
    dtype(Time)

To coerce values to a specific type, specify the dtype explicitly.

    >>> arr = np.array(["2018-01-01", "2018-07-04", "2018-07-11"], dtype=Date)
    >>> arr
    array([Date(2018, Jan, 1), Date(2018, Jul, 4), Date(2018, Jul, 11)],
          dtype=Date)

You can use arithmetic and NumPy's built-in ufuncs.

    >>> np.full(8, Date(2018, 1, 1)) + np.arange(8)
    array([Date(2018, Jan, 1), Date(2018, Jan, 2), Date(2018, Jan, 3),
           Date(2018, Jan, 4), Date(2018, Jan, 5), Date(2018, Jan, 6),
           Date(2018, Jan, 7), Date(2018, Jan, 8)], dtype=Date)


Casting
-------

You can cast a NumPy `datetime64` array with units "s" or smaller to an Ora time
type.

    >>> arr
    array(['2000-01-01T00:00:00.000000000', '2019-05-27T18:44:26.000000000'],
          dtype='datetime64[ns]')
    >>> arr.astype(Time)
    array([ora.Time(2000, 1, 1, 0, 0, 0.00000000, UTC),
           ora.Time(2019, 5, 27, 18, 44, 26.00000000, UTC)], dtype=Time)

You can also cast in the ohter direction.

    >>> arr
    array([ora.Time(2019, 5, 27, 18, 45, 24.13385701, UTC),
           ora.Time(2019, 5, 27, 18, 45, 24.13386100, UTC),
           ora.Time(2019, 5, 27, 18, 45, 24.13386100, UTC),
           ora.Time(2019, 5, 27, 18, 45, 24.13386198, UTC)], dtype=Time)
    >>> arr.astype("datetime64[ns]")
    array(['2019-05-27T18:45:24.133857012', '2019-05-27T18:45:24.133861005',
           '2019-05-27T18:45:24.133861005', '2019-05-27T18:45:24.133861989'],
          dtype='datetime64[ns]')

When casting, Ora always assumes that the datetime64 array uses UTC.

You can likewise cast a NumPy `datetime64[D]` array to and from an Ora date
dtype.

    >>> arr = Date(2019, 5, 27) + np.arange(3)
    >>> arr
    array([Date(2019, May, 27), Date(2019, May, 28), Date(2019, May, 29)],
          dtype=Date)
    >>> arr.astype("datetime64[D]")
    array(['2019-05-27', '2019-05-28', '2019-05-29'], dtype='datetime64[D]')

Ora will not let you cast a `datetime64[D]` array to or from a time array, nor
will it let you cast a `datetime64` with unit seconds or smaller to or from a
date array.  In either case, Ora will set all values in the result to `INVALID`
for a date or time dtype, or `NAT` for a `datetime64` dtype.


Functions
---------

The `ora.np` module contains ufuncs and NumPy-specific functions.  For instance,
to construct a date from components,

    >>> year = [1988, 2001, 2018]
    >>> month = [10, 12, 7]
    >>> day = [23, 3, 11]
    >>> arr = ora.np.date_from_ymd(year, month, day)
    >>> arr
    array([Date(1988, Oct, 23), Date(2001, Dec, 3), Date(2018, Jul, 11)],
          dtype=Date)

The inverse function returns a structured array.

    >>> ymd = ora.np.get_ymd(arr)
    >>> ymd
    array([(1988, 10, 23), (2001, 12,  3), (2018,  7, 11)],
          dtype=[('year', '<i2'), ('month', 'u1'), ('day', 'u1')])
    >>> ymd["year"]
    array([1988, 2001, 2018], dtype=int16)

You can localize an array of times to a time zone.

    >>> arr = np.array([now(), now(), now()])
    >>> date, daytime = ora.np.to_local(arr, "America/New_York")
    >>> date
    array([Date(2018, Jul, 11), Date(2018, Jul, 11), Date(2018, Jul, 11)],
          dtype=Date)


Internals
---------

The NumPy array stores the underlying time, date, or daytime integer offset
directly, similar to NumPy's own `datetime64` dtypes.

    >>> Date.dtype.itemsize
    4
    >>> Time.dtype.itemsize
    8
    >>> Daytime.dtype.itemsize

Use `ora.np.to_offset()` to obtain an array of the underlying integer offsets.  

This function creates a new array containing offsets.  Since the offset is the
internal representation of a time, you can obtain a similar array, albeit with
shared array data, using the ndarray `view()` method and the integer type
corresponding to the Ora date, time, or daytime type.




API
---

This section lists the functions and ufuncs that operate on arrays with Ora
dtypes.  These are available in the `ora.np` module.

Functions
^^^^^^^^^

These functions produce NumPy arrays of Ora objects.  

.. function:: date_from_ordinal_date(year, ordinal)

    Constructs dates from years and ordinal dates, like
    `Date.from_ordinal_date`.

.. function:: date_from_week_date(week_year, week, weekday)

    Constructs dates from ISO week days, like `Date.from_week_date`.

.. function:: date_from_ymd(year, month, day)

    Contructs dates from day, month, and year components, like
    `Date.from_ymd`.

.. function:: date_from_ymdi(ymdi)

    Constructs dates from YYYYMMDD integers, like `Date.from_ymdi`.

.. function:: time_from_offset(offset)

    Constructs times from number of ticks, like `Time.from_offset`.  The
    duration of a tick, and the epoch time from which it's measured, depends
    on the Ora time type.

.. function:: to_local(time, time_zone)

    Converts times to local dates and daytimes in a given time zone, like
    `ora.to_local`.  Returns a date array and a daytime array.

.. function:: from_local(date, daytime, time_zone)

    Converts local dates and daytimes to times in a given time zone, like
    `ora.from_local`.  Returns a time array.

The functions above also accept `Time`, `Date`, and/or `Daytime` keyword
arguments, to control the dtypes of the resulting arrays.

    >>> ora.np.time_from_offset(np.arange(4), Time=Unix32Time)
    array([ora.Unix32Time(1970, 1, 1, 0, 0, 0., UTC),
           ora.Unix32Time(1970, 1, 1, 0, 0, 1., UTC),
           ora.Unix32Time(1970, 1, 1, 0, 0, 2., UTC),
           ora.Unix32Time(1970, 1, 1, 0, 0, 3., UTC)], dtype=Unix32Time)

Ufunc-style broadcasting is applied to the arguments.

    >>> ora.np.date_from_ymd(2019, [1, 2], [[3, 4], [5, 6]])
    array([[Date(2019, Jan, 3), Date(2019, Feb, 4)],
           [Date(2019, Jan, 5), Date(2019, Feb, 6)]], dtype=Date)


Ufuncs
^^^^^^

.. function:: is_valid(obj)

    Returns a boolean array indicating true where the value is valid.  Works on
    time, date, and daytime arrays.

.. function:: to_offset(obj)

    Returns the offset (ticks) of the time, date, or daytime array.  The offset
    dtype depends on the dtype of the argument.  Each Ora type uses a specific
    signed or unsigned integer to represent its offset.

.. function:: get_day(date)
.. function:: get_month(date)
.. function:: get_ordinal_date(date)
.. function:: get_week_date(date)
.. function:: get_weekday(date)
.. function:: get_year(date)
.. function:: get_ymd(date)
.. function:: get_ymdi(date)



Dtypes
^^^^^^

- ORDINAL_DATE_DTYPE <class 'numpy.dtype'>
- WEEK_DATE_DTYPE <class 'numpy.dtype'>
- YMD_DTYPE <class 'numpy.dtype'>
