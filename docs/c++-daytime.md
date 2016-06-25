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


