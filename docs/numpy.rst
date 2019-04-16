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

The NumPy array stores the underlying time or date integer offset directly,
similar to NumPy's own `datetime64` dtypes.

    >>> Date.dtype.itemsize
    4
    >>> Time.dtype.itemsize
    8

Use `ora.np.to_offset()` to obtain an array of the underlying integer offsets.  
You can also array `view()` method.


API
---

This section lists the functions and ufuncs that operate on arrays with Ora
dtypes.  These are available in the `ora.np` module.

Functions
^^^^^^^^^

.. function:: date_from_ordinal_date(year, ordinal)

    Constructs dates from years and ordinal dates.  Broadcasting rules apply.


- date_from_week_date <class 'builtin_function_or_method'>
- date_from_ymd <class 'builtin_function_or_method'>
- date_from_ymdi <class 'builtin_function_or_method'>
- from_local <class 'builtin_function_or_method'>
- from_offset <class 'builtin_function_or_method'>


Ufuncs
^^^^^^

- get_day <class 'numpy.ufunc'>
- get_month <class 'numpy.ufunc'>
- get_ordinal_date <class 'numpy.ufunc'>
- get_week_date <class 'numpy.ufunc'>
- get_weekday <class 'numpy.ufunc'>
- get_year <class 'numpy.ufunc'>
- get_ymd <class 'numpy.ufunc'>
- get_ymdi <class 'numpy.ufunc'>
- is_valid <class 'numpy.ufunc'>
- numpy <class 'module'>
- to_local <class 'builtin_function_or_method'>
- to_offset <class 'numpy.ufunc'>


Dtypes
^^^^^^

- ORDINAL_DATE_DTYPE <class 'numpy.dtype'>
- WEEK_DATE_DTYPE <class 'numpy.dtype'>
- YMD_DTYPE <class 'numpy.dtype'>
