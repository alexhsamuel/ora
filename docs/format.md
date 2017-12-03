# Format codes

`cron::Format` uses a format pattern similar to that of `strftime()` to specify
the format of dates and times.

The following format codes are supported for dates and times:

| Code | Expansion |
|:----:|:----------|
| `%b` | the month name |
| `%d` | the one-indexed day of the month |
| `%D` | (not implemented) |
| `%g` | the last two digits of the week number of the week date |
| `%G` | the full week year of the week date |
| `%j` | the one-indexed ordinal day of the year |
| `%m` | the one-indexed month number |
| `%V` | the one-indexed week number of the year |
| `%w` | the weekday number, 0 = Sunday through 6 = Saturday |
| `%W` | the weekday name |
| `%y` | the last two digits of the year |
| `%Y` | the full year |

The following format codes are supported for daytimes and times:

| Code | Expansion |
|:----:|:----------|
| `%h` | the 12-hour hour number |
| `%H` | the 24-hour hour number |
| `%M` | the minute number |
| `%p` | either "AM", for hour < 12, or "PM", otherwise |
| `%S` | seconds of the minute |
| `%T` | (not implemented) |

The following format codes are supported for times only:

| Code | Expansion |
|:----:|:----------|
| `%o` | the total signed time zone offset in seconds |
| `%q` | the minutes part of the time zone offset |
| `%Q` | the hours part of the time zone offset |
| `%U` | the sign of the time zone offset |
| `%Z` | the time zone name |
| `%z` | the [military time zone](https://en.wikipedia.org/wiki/List_of_military_time_zones) letter |

The following modifiers are supported:

 * A numerical code can be preceded by the number of (integral) digits to show.
   For example, `%3h` uses three digits for the number of hours,

 * For `S`, the number of digits may be followed by a decimal point and number
   of fractional digits.  For example, `%02.3S` shows seconds to ms precision.

 * `#` followed by another character sets the pad character.  For example,
   `%#*3H` shows the number of hours with three digits, padded on the left with
   asterisks.

 * `^` specifies all capital letters, for example `%^W`.

 * `_` specifies all lower-case letters, for example `%_b`.

 * `~` specifies abbreviated names, for months, weekdays, and time zones, for
   example `%~W`.

 * `E` (not implemented) specifies the locale's alternative representation.

 * `O` (not implemented) specifies the locale's alternative numerical
   representation.


