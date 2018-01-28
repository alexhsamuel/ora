# Format codes

`ora::Format` uses a format pattern similar to that of `strftime()` to specify
the format of dates and times.

The following format codes are supported for dates and times:

| Code | Expansion |
|:----:|:----------|
| `%A` | the weekday name |
| `%a` | the weekday abbreviated name; same as `%~A` |
| `%B` | the month name |
| `%b` | the month abbreviated name; same as `%~B` |
| `%D` | the date as "YYYY-MM-DD", or "YYYYMMDD" for `%~D` |
| `%d` | the one-indexed day of the month |
| `%g` | the last two digits of the week year of the ISO week date |
| `%G` | the week year of the ISO week date |
| `%j` | the one-indexed ordinal day of the year |
| `%m` | the one-indexed month number |
| `%V` | the one-indexed week number of the ISO week date |
| `%w` | the weekday number, 0 = Sunday through 6 = Saturday |
| `%u` | the weekday number of the ISO week date, 1 = Monday through 7 = Sunday |
| `%y` | the last two digits of the year |
| `%Y` | the full year |

The following format codes are supported for daytimes and times:

| Code | Expansion |
|:----:|:----------|
| '%f' | 6 fractional digits (truncated) of seconds |
| `%H` | the 24-hour hour number |
| `%I` | the 12-hour hour number |
| `%M` | the minute number |
| `%p` | either "AM", for hour < 12, or "PM", otherwise |
| `%S` | seconds of the minute |

The following format codes are supported for times only:

| Code | Expansion |
|:----:|:----------|
| `%E` | the six-character time zone offset, e.g. +00:00 |
| `%e` | the [military time zone](https://en.wikipedia.org/wiki/List_of_military_time_zones) letter |
| `%i` | the ISO 8601 time format |
| `%o` | the total signed time zone offset in seconds |
| `%q` | the minutes part of the time zone offset |
| `%Q` | the signed hours part of the time zone offset, e.g. +00 |
| `%Z` | the time zone name |
| `%z` | the five-character time zone offset, e.g. +0000 |

The following modifiers are supported:

- A numerical code can be preceded by the number of (integral) digits to show.
  For example, `%3h` uses three digits for the number of hours,

- For `S`, the number of digits may be followed by a decimal point and number
  of fractional digits.  For example, `%02.3S` shows seconds to ms precision.

- `#` followed by another character sets the pad character.  For example,
  `%#*3H` shows the number of hours with three digits, padded on the left with
  asterisks.

- `^` specifies all capital letters, for example `%^W`.

- `_` specifies all lower-case letters, for example `%_b`.

- `~` specifies abbreviated names, for months, weekdays, and time zones, for
   example `%~W`.

- `&` (not implemented) specifies the locale's alternative representation.

- `$` (not implemented) specifies the locale's alternative numerical
  representation.


## Time zones

The `TimeFormat` C++ class, and the Python formatting strings shown above,
always format times localized to UTC.

The `LocalTimeFormat` C++ class and Python formatting for times support an
additional syntax for specifying the time zone in which times are formatted.
The time format may be followed by `@` and a time zone specification; these
specifications are supported:

- `display` or empty string: the current display time zone
- `system`: the system time zone
- a time zone name: the named time zone

For example, the format string `"%Y-%m-%d %H:%M:%S%E@America/New_York"` formats
the time, with UTC offset, localized to New York.

