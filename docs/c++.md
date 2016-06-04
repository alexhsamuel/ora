# C++ API

## Dates

```c++
#include "cron.hh"

using cron::date;
```

A `cron::date::Date` represents a calendar date.  A calendar date represents a period, usually (but not always) 24 hours long, in a specific location; as such, a date does not by itself correspond to any time or interval of times.

### Building dates

Cron provides various date factory functions. The most common convention for specifying a date is the _YMD_ form, a triplet of year number, month number, and day of month.  

```c++
Date date = from_ymd(1973, 12, 3);
```

It is a sloppy but common convention to encode the year, month, and day into eight decimal digits of a 32-bit integer, which cron calls _YMDI_.

```c++
auto date = from_ymdi(19731203);
```

Another date representation is the _ordinal date_, which specifies the year and the (one-indexed) day of the year.  Yet another is the (ISO) _week date_, which specifies the year, week number, and day of week.  The year in the week date may be different than the year in the YMD or ordinal representations.  The week number is counted from Monday = 0 through Sunday = 6; constants are provided for these.

```c++
auto date = from_ordinal(1973, 337);
auto date = from_week_date(1973, 48, MONDAY);
```


#### Date literals

The `cron::ez` namespace adds syntactic sugar for specifying date literals.

```c++
using namespace cron::ez;

auto date = 1973/DEC/3;
```

Special three-letter month constants must be used (`JAN`, `FEB`, ...), and leading zeros are not allowed for the year and day.  Such literals are `constexpr` and may be used as compile-time constants.


### Date accessors

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
Year year       = get_year(date);
Month month     = get_month(date);
Day day         = get_day(date);
Ordinal ordinal = get_ordinal(date);
Year week_year  = get_week_year(date);
Week week       = get_week(date);
Weekday weekday = get_weekday(date);
```

Note that the year (of the YMD and ordinal representations) shares a _type_ with the "week year" of the week date, but they may have different values.


### Shifting dates

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


### Date representations

`Date` supports dates between 0001-01-01 (Jan 1 of the year 1 C.E.) and 9999-12-31 (Dec 31, 9999).  

An instance stores the date as an `uint32_t` offset from 0001-01-01, and may efficiently be passed by value.  It has no virtual methods or other state, so `uint32_t*` may safely be cast to and from `Date*`.

```c++
int offset = date.get_offset();
Date* alias = (Date*) &offset;
```

Cron also provides a 16-bit `Date16` class, which stores dates as `uint16_t` offsets from 1970-01-01; the last representable date is 2149-06-04.  All the factory functions accept the date type as a template argument.

```c++
Date16 date = from_ymd<Date16>(1973, 12, 3);
```

Each date class has `MIN` and `MAX` static attributes containing the earliest and latest representable dates.  Other than this, the date classes have identical APIs.

The various dates are mutually constructible and assignable, as long as the actual dates are representable.

```c++
Date date = from_ymd<Date16>(1973, 12, 3);
Date16 date = 1973/DEC/3;  // from namespace cron::ez
```
