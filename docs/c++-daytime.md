# C++ Daytime API

```c++
#include "cron.hh"

using cron::daytime;
```

A `cron::daytime::Daytime` represents an approximate time of day.  Daytime is a representation of a specific time within a specific day in a specific location, as one might read off an ordinary clock.

See [c++-localization](c++-localization.md) for how to use a daytime to build time values.


## Buildling daytimes

Cron provides several daytime factory functions. The most common convention for specifying a daytime is the _HMS_ form, a triplet of hour, minute, and second. 

```c++
Daytime daytime;
daytime = from_hms(10, 30, 0.0);
daytime = from_hms(10, 30);       // same thing; seconds defaults to 0
```

Another daytime representation is _SSM_ (_seconds since midnight_), the number of seconds that have passed since the previous midnight.

```c++
auto const noon = from_ssm(43200.0); 
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

