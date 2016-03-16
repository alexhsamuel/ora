This document describes the Python API to Cron.  Please see [ontology.md](ontology.md) for a description of the main concepts and terms in Cron, and their semantics.

The code below assumes the following import, but of course feel free to use qualified names in your programs.

```py
>>> from cron import *
```

# Types

The C++ types for times, dates, and daytimes are templated; Python, however, does not support templates, as they are a compile-time construct.  Instead, The Python extension module contains a variety of independent extension types that wrap various instances of the C++ templates.  Each of these is an independent type, but with the same API as the other variants.

## Dates

`Date` is the default type for dates.

```py
>>> d = Date(2016, 3, 15)
>>> print(d)
2016-03-15
```

The `SmallDate` class is similar, but uses a 16-bit integer internally, and therefore has a narrower range of dates it can represent.

### Date parts

The components of date representations are available as attributes.  These include the default representation, as well as the ordinal date and week date representations.

```py
>>> d.year, d.month, d.day
(2016, Month.Mar, 15)
>>> d.year, d.ordinal
(2016, 75)
>>> d.week_year, d.week, d.weekday
(2016, 10, Weekday.Tue)
```

These components are also accessible in the `parts` attribute, whose value can be unpacked to produce the ordinary date components, but which also has the other components as attributes.  

<font color="red">FIXME:</font> Should we dispents with `parts` and just make the date itself iterable?

```py
>>> year, month, day = d.parts
>>> d.parts.weekday
Weekday.Tue
```

There's also a `ymdi` attribute, which contains the date parts encoded in an eight-digit decimal integer.

```py
>>> d.ymdi
20160315
```

### Date literals

Months and weekdays are both given as enumerations, respectively `Month` and `Weekday`.  The enumerals have three-letter abbreviated names.

```py
>>> Thu
Weekday.Thu
>>> Oct
Month.Oct
```

The month enumerals also define the `__truediv__` operator to provide this syntactic trick for writing date literals:

```py
>>> 2016/Mar/15
Date(2016, Mar, 15)
```

### Date conversion

The `Date` constructor makes an effort to convert reasonable date representations to the date type.  These include,

- Instances of other Cron date classes.
- Python `datetime.date` instances.
- An integer between 10000000 and 99999999  is interpreted as a YMDI date.
- A three-element sequence is interpreted as a (year, month, day) triplet.
- A two-element sequence is interpreted as a (year, ordinal) pair.

For example,

```py
>>> Date(SmallDate(2016, Mar, 15))
Date(2016, Mar, 15)
>>> Date(datetime.date(2016, 3, 15))
Date(2016, Mar, 15)
>>> Date(20160315)
Date(2016, Mar, 15)
>>> Date((2016, 3, 15))
Date(2016, Mar, 15)
>>> Date([2016, 75])
Date(2016, Mar, 15)
```

Most Cron functions that take a date parameter will accept any of these.

### Special dates

Each date class provides `MIN` and `MAX` attributes, giving the earliest and latest representable dates.

```py
>>> print(Date.MIN, Date.MAX)
0001-01-01 9999-12-31
>>> print(SmallDate.MIN, SmallDate.MAX)
1970-01-01 2149-06-04
```

Each class also provides two special date values, distinct from all other dates.

```py
>>> Date.INVALID
Date.INVALID
>>> Date.MISSING
Date.MISSING
```

Attributes `invalid` and `missing` test for these, and `valid` is true iff. the date is neither invalid nor missing.

```py
>>> d.valid, d.invalid, d.missing
(True, False, False)
```

The two special values compare false to date, including themselves.  
The `is_same()` method, however, compares two dates, including missing and invalid dates.

```py
>>> Date.INVALID == Date.INVALID, 
False
>>> Date.INVALID.is_same(Date.INVALID)
True
```

### Arithemtic

Adding or subtracting from a date shifts the date forward or backward.

```py
>>> print(d + 10)
2016-03-25
>>> print(d - 10)
2016-03-05
```

The difference between two dates is the number of days between them.

```py
>>> d - 2016/Jan/1
74
```

Note that this value is one smaller than the date's ordinal, 75, since the ordinal is one-indexed.

