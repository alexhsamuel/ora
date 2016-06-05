# C++ Date API

```c++
include "cron.hh"

using cron::date;
```

A `cron::date::Date` represents a calendar date.  A calendar date represents a period, usually (but not always) 24 hours long, in a specific location; as such, a date does not by itself correspond to any time or interval of times.

## Building dates

Cron provides various date factory functions. The most common convention for specifying a date is the _YMD_ form, a triplet of year number, month number, and day of month.  

```c++
Date date;
date = from_ymd(1973, 12, 3);
```

It is a sloppy but common convention to encode YMD into eight decimal digits of a 32-bit integer, which cron calls _YMDI_.

```c++
auto date = from_ymdi(19731203);
```

Another date representation is the _ordinal date_, which specifies the year and the (one-indexed) day of the year.  Yet another is the (ISO) _week date_, which specifies the year, week, and day of week.  

```c++
auto date = from_ordinal(1973, 337);
auto date = from_week_date(1973, 48, MONDAY);
```

A few things to keep in mind about various date components:

- Month, day, ordinal, and week are (according to tradition rather than reason) one-indexed.
- Weekdays are counted from `MONDAY` = 0 through `SUNDAY` = 6; constants are provided for these.
- The YMD and ordinal date representations share the same year, but the year in the week date representation may be different.

If the arguments you provide are invalid, cron throws `InvalidDateError`.


### Date literals

The `cron::ez` namespace adds syntactic sugar for specifying date literals.

```c++
using namespace cron::ez;
auto date = 1973/DEC/3;
```

Special three-letter month constants must be used (`JAN`, `FEB`, ...), and leading zeros are not allowed for the year and day.  

## Date accessors

Functions are provided to produce the various date representations.  These return structures with the relevant components.

```c++
auto ymd = get_ymd(ymd);
std::cout << ymd.month << "/" << ymd.day << "/" << ymd.year << "\n";

auto ordinal_date = get_ordinal_date(date);
auto week_date = get_week_date(date);
auto ymdi = get_ymdi(date);
```

There are also functions to return individual components directly, with a type alias for each component.

```c++
Year year       = get_year(date);       // for ymd and ordinal dates 
Month month     = get_month(date);
Day day         = get_day(date);
Ordinal ordinal = get_ordinal(date);
Year week_year  = get_week_year(date);  // not same as get_year()!
Week week       = get_week(date);
Weekday weekday = get_weekday(date);
```


## Shifting dates

The `days_after()` and `days_before()` functions shift a date forward or backward by some number of calendar days.  Negative shifts may be used; two functions are provided only for convenience.

```c++
auto next_week = days_after(date, 7);
auto last_week = days_before(date, 7);  // or days_after(date, -7)
```

The addition and subtraction operators are overloaded as synonyms of these two, respectively.

```c++
auto next_week = date + 7;
auto last_week = date - 7;  // or date + -7
```

`days_between(date0, date1)` returns the number of days by which `date0` must be shifted forward to arrive at `date1`.  Subtraction of two dates is equivalent, _but with the order of arguments swapped_.

```c++
int days_ago = days_between(past_date, today);
int days_ago = today - past_date;
```


## Date representations

`Date` supports dates between 0001-01-01 (Jan 1 of the year 1 C.E.) and 9999-12-31 (Dec 31, 9999).  

An instance stores the date as an `uint32_t` offset from 0001-01-01, and may efficiently be passed by value.  It has no virtual methods or other state, so `uint32_t*` may be cast to and from `Date*`, as long as the value corresponds to a valid offset.

```c++
int offset = date.get_offset();
Date* alias = (Date*) &offset;
```

Cron also provides a 16-bit `Date16` class, which stores dates as `uint16_t` offsets from 1970-01-01; the last representable date is 2149-06-04.  All the factory functions accept the date type as a template argument.

```c++
Date16 date = from_ymd<Date16>(1973, 12, 3);
```

Each date class has `MIN` and `MAX` static attributes containing the earliest and latest representable dates.  Other than this, the date classes have identical APIs.

The various dates are mutually conversion-constructible and -assignable, as long as the actual dates are representable.

```c++
using namespace cron::ez;
Date16 date = 1973/DEC/3;  // RHS is Date, so convert
```

If you try to store a date in `Date16` that is outside the representable range, cron throws `DateRangeError`.

```c++
try {
  Date16 date = 9999/Dec/31;
}
catch (DateRangeError err) {
  // Oops!
}
```


## Invalid dates

Each date class provides two special values.

- `INVALID` represents an uninitialized date or the result of a failed operation.
- `MISSING` is a placeholder that you can use to represent a value that is not available; it is never produced by cron itself.


```c++
Date date;  // default ctor initializes to INVALID
date = Date::INVALID;
date = Date::MISSING;
```

The `is_invalid()` and `is_missing()` methods test for these two; `is_valid()` is true iff. the date is neither.

```
if (date.is_valid())
  std::cout << date;
else
  std::cout << "something's wrong!";
```

If you call any function on a missing or invalid date, cron throws `InvalidDateError`.

```c++
Date date;  // default ctor initializes to INVALID
try {
  std::cout << get_year(date) << "\n";
}
catch (InvalidDateError err) {
  // Oops!
}
```


## Safe functions

The `cron::date::safe` namespace provides alternatives to all date functions that don't throw exceptions; instead, they return special values to indicate failure.

- Any function that returns a date will return `INVALID` instead.

  ```c++
auto date = safe::from_ymd(1980, 2, 31);    // no such thing as Feb 31 -> INVALID
date = safe::days_after(Date::MISSING, 1);  // can't shift -> INVALID
```

- Accessors will return special invalid values instead.

  ```c++
auto year = safe::get_year(Date::INVALID);  // -> YEAR_INVALID
auto ymdi = safe::get_ymdi(Date::MISSING);  // -> YMDI_INVALID
```

