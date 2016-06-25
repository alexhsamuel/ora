# C++ Daytime API

```c++
#include "cron.hh"
using namespace cron;
```

A `cron::daytime::Daytime` represents an approximate time of day.  Daytime is a representation of a specific time within a specific day in a specific location, as one might read off an ordinary clock.

See [c++-localization](c++-localization.md) for how to use a daytime to build time values.


## Buildling daytimes

Cron provides several daytime factory functions. The most common convention for specifying a daytime is the _HMS_ form, a triplet of hour, minute, and second. 

```c++
daytime::Daytime daytime;
daytime = daytime::from_hms(10, 30, 0.0);
daytime = daytime::from_hms(10, 30);       // same thing; seconds defaults to 0
```

Another daytime representation is _SSM_ (_seconds since midnight_), the number of seconds that have passed since the previous midnight.

```c++
auto const noon = daytime::from_ssm(43200.0); 
```

Keep in mind:

- `Daytime` represents what you'd read on a clock; it is not aware of DST transitions. It considers a day equal to exactly 24 full hours, or 86400 seconds.
- `Hour` must be between 0 and 23.
- `Minute` must be between 0 and 59.
- `Second` is a floating point value at least 0 and strictly less than 60.


## Daytime accessors

Cron provides functions to compute the various daytime representations from a `Daytime` object.  `get_hms` returns a structure with hour, minute, and second components.

```c++
auto hms = get_hms(daytime);
std::cout << hms.hour << "hr, " << hms.minute << "min, " << hms.second << "sec\n";
```

There are also functions to return individual components, as well as SSM.  Cron provides a type alias for each.

```c++
Hour hour       = get_hour(daytime);
Minute minute   = get_minute(daytime);
Second second   = get_second(daytime);

Ssm ssm         = get_ssm(daytime);
```


# Shifting daytimes

The `seconds_after()` and `seconds_before()` functions shift a daytime forward or backward by some number of seconds.  Negative shifts may be used; two functions are provided only for convenience.  Shifted daytimes are computed modulo a standard 24-hour day.

```c++
auto in_an_hour   = seconds_after(now, SECS_PER_HOUR);
auto an_hour_ago  = seconds_before(now, SECS_PER_HOUR);

assert now == seconds_after(now, SECS_PER_DAY);  // no-op
```

The addition and subtraction operators are overloaded as synonyms of these two, respectively.

```c++
auto tock = tick + 1;  // one second later
```

`seconds_between(daytime0, daytime1)` returns the number of seconds by which `daytime0` must be shifted forward to arrive at `daytime1`.  The result is positive iff. `daytime0` precedes `daytime1` assuming the same day.


## Daytime representations

The default daytime format is the [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) _HH:MM:SS_ format.

### Formatting

Cron provides overloads for `to_string` and `operator<<` that render a daytime in the default format.

`DaytimeFormat` provides flexible formatting of daytimes.  An instance takes an extended [strftime](http://man7.org/linux/man-pages/man3/strftime.3.html)-style format string; see [format.md](format.md) for codes.  Notably, you may specify `printf`-style precision with `%S` to show fractional seconds, for example `%.5S` for five decimal places of precision.

`DaytimeFormat::operator()` formats a daytime.

```c++
DaytimeFormat fmt("%h:%M:%.1S %p");
std::cout << fmt(from_hms(23, 45, 6.78));  // prints '11:45:06.7 PM'
```


### Parsing

Not implemented yet.


## Internals

`Daytime` supports daytimes between midnight and the following midnight, inclusive of the former.  The resolution is `Daytime::RESOLUTION`, around 7 fs.  The largest representable daytime is a very small amount of time before midnight.


### Storate representation

An instance stores the daytime as an `uint64_t` offset from midnight, and may efficiently be passed by value.  It has no virtual methods or other state, so `uint64_t*` may be cast to and from `Daytime*`, as long as the value corresponds to a valid offset.


### Alternate daytime classes

Cron also provides a 32-bit `Daytime32` class, which stores dates as `uint32_t`, with a resolution of about 31 µs.  All the factory functions accept the daytime type as a template argument.

```c++
Daytime32 date = from_hms<Daytime32>(12, 30);
```

Each daytime class has `MIN` and `MAX` static attributes containing the earliest and latest representable dates.  Other than this, the date classes have identical APIs.

The daytime types are mutually conversion-constructible and -assignable, as long as the actual dates are representable.

```c++
Daytime32 date = from_hms(12, 30);  // RHS is Daytime, so convert
```


## Invalid daytimes

Each daytime class provides two special values.

- `INVALID` represents an uninitialized daytime or the result of a failed operation.
- `MISSING` is a placeholder that you can use to represent a value that is not available; it is never produced by cron itself.


```c++
Daytime daytime;  // default ctor initializes to INVALID
daytime = Daytime::INVALID;
daytime = Daytime::MISSING;
```

The `is_invalid()` and `is_missing()` methods test for these two; `is_valid()` is true iff. the daytime is neither.

```
if (daytime.is_valid())
  std::cout << daytime;
else
  std::cout << "something's wrong!";
```

If you call a function on a missing or invalid daytime, cron throws `InvalidDaytimeError`.

```c++
Daytime daytime;  // default ctor initializes to INVALID
try {
  std::cout << get_year(daytime) << "\n";
}
catch (InvalidDaytimeError err) {
  // Oops!
}
```

### Comparisons

The usual equality and ordering operators work with invalid and missing daytimes.  The order is,

```c++
Daytime::INVALID < Daytime::MISSING < Daytime::MIN < ... < Daytime::MAX
```


## Safe functions

The `cron::daytime::safe` namespace provides "safe" alternatives to all daytime functions, which don't throw exceptions; instead, they return special values to indicate failure.

- Any function that returns a daytime will return `INVALID` instead.

  ```c++
auto daytime = safe::from_hms(30, 0, 0);    // no such thing as 30 hours -> INVALID
daytime = safe::seconds_after(Daytime::MISSING, 1);  // can't shift -> INVALID
```

- Accessors will return special invalid values instead.

  ```c++
auto hour = safe::get_hour(Daytime::INVALID);  // -> HOUR_INVALID
```

These safe variants are particularly useful when you are vectorizing daytime operations and don't want individual failures to throw you out of a loop.

