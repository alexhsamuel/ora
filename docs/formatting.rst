Formatting
==========

Ora supports formatting using format specifiers that are a superset of Python's
built-in strftime/strptime-style `formatting mini language
<https://docs.python.org/3.6/library/datetime.html#strftime-and-strptime-behavior>`_.
(Note that locale-specific formatting codes, `%U`/`%W`, and `%Z` are not
implemented yet.)

These formatting specifications are available with `format()`, `str.format()`,
and "f-string" interpolation.

    >>> format(date, "%B %d, %Y")
    'July 12, 2018'
    >>> "Today is {:%A}.".format(date)
    'Today is Thursday.'
    >>> f"now: {time:%i}"
    'now: 2018-07-12T15:36:08+00:00'

For times, dates, and daytimes, the default format is the ISO 8601 format, with
decimial digits appropriate for the type.  This is also the `str` format.


Codes
-----

The following table lists formatting codes.

==== ============================ ===========
Code Example                      Description 
==== ============================ ===========
`%A` `Friday`                     weekday name 
`%a` `Fri`                        weekday abbreviated name 
`%B` `March`                      the month name 
`%b` `Mar`                        the month abbreviated name 
`%C` `23:59:59`                   ISO 8601 daytime representation 
`%D` `2018-07-11`                 ISO 8601 date (without hyphens, if abbreviated) 
`%d` `11`                         the one-indexed day of the month 
`%E` `-05:00`                     the six-character UTC offset, e.g. +00:00 
`%e` `Z`                          the `military time zone <https://en.wikipedia.org/wiki/List_of_military_time_zones>`_ letter 
`%f` `012345`                     6 fractional digits (truncated) of seconds 
`%G` `2018`                       ISO 8601 week year 
`%g` `18`                         the last two digits of the week number of the week date 
`%H` `23`                         the 24-hour hour number 
`%I` `11`                         the 12-hour hour number 
`%i` `2018-07-11T12:30:36+00:00`  ISO 8601 time (without hyphens and colons, if abbreviated) 
`%j` `092`                        the one-indexed ordinal day of the year 
`%M` `08`                         the minute number 
`%m` `07`                         the one-indexed month number 
`%o` `-18000`                     the total signed UTC offset in seconds 
`%p` `AM`                         either "AM", for hour < 12, or "PM", otherwise 
`%S` `08`                         seconds of the minute 
`%T` `2018-07-11T12:30:36Z`       ISO 8601 time with military time zone letter 
`%u` `3`                          ISO 8601 weekday as a decimal number, 1 = Mon through 7 = Sun 
`%V` `28`                         ISO 8601 week as a decimal number 
`%w` `5`                          the weekday number, 0 = Sun through 6 = Sat 
`%Y` `2018`                       the full year 
`%y` `18`                         the last two digits of the year 
`%z` `-0500`                      5-char UTC offset, e.g. +0000 
==== ============================ ===========


Modifiers
---------

Ora understands modifiers, which appear after % and before the code letter.

**Digits**

A numerical code can be preceded by the number of (integral) digits to show.
For example, `%3h` uses three digits for the number of hours,

For `%S`, the number of digits may be followed by a decimal point and number of
fractional digits.  For example, `%02.3S` shows seconds to ms precision.

For `%i` and `%T`, a decimal point and the number specify the number of
fractional digits.

    >>> format(time, "%i")
    '2018-07-12T15:28:53+00:00'
    >>> format(time, "%.6i")
    '2018-07-12T15:28:53.329243+00:00'

**Padding**

`#` followed by another character sets the pad character.  For example, `%#*3H`
shows the number of hours with three digits, padded on the left with asterisks.

    >>> format(time, "%#*4H %#*4M %#*4S")
    '**15 **36 ***8'

**Capitalization**

For month and weekday names, `^` specifies all capital letters, and `\_`
specifies all lower-case letters.

    >>> format(time, "%^A")
    'THURSDAY'
    >>> format(time, "%_A")
    'thursday'

**Abbreviation**

For ISO formats (`%i`, `%T`, `%C`, `%D`), the modifier `\~` omits hyphens and
colons.

      >>> format(time, "%i")
      '2018-07-12T15:28:53+00:00'
      >>> format(time, "%~i")
      '20180712T152853+0000'

For `%A` and `%B`, `\~` selects the abbreviated name, like `%a` and `%b`
respectively.


Time zones
----------

By default, times are formatted in UTC.  To use a different time zone, follow
the format string with `@` and a time zone name.

    >>> format(time, "%i")
    '2018-07-12T16:30:20+00:00'
    >>> format(time, "%i@America/New_York")
    '2018-07-12T12:30:20-04:00'

With `str.format()` and interpolated strings, Python allows you to specify the
time zone name with another substitution.

    >>> f"{time:%i@{time_zone}}"
    '2018-07-12T12:30:20-04:00'

You can ommit the format code entirely and specify only a time zone, if you want
the ISO 8601 format.

    >>> format(time, "@America/New_York")
    '2018-07-12T12:30:20-04:00'

You can also specify "display" or "system" as the time zone name; see `Display
time zone` and `System time zone`.  If you omit the time zone name, Ora assumes
"display".  So, you can format a time in the display time zone rather than UTC
by appending `@` to the format.

    >>> format(time)
    '2018-07-12T16:30:20+00:00'
    >>> format(time, "@")
    '2018-07-12T11:30:20-05:00'
    >>> get_display_time_zone()
    TimeZone('America/Chicago')


Parsing
=======

Ora provides functions to parse times, dates, and daytimes using the same format
strings.

    >>> parse_time("%i", "2018-07-12T11:30:20-05:00")
    ora.Time(2018, 7, 12, 16, 30, 20.00000000, UTC)

    >>> parse_date("%B %d, %Y", "July 11, 2018")
    Date(2018, Jul, 11)

    >>> parse_daytime("%H:%M", "12:30")
    Daytime(12, 30, 00.000000000000000)

A time object represents a physical time, and a date and daytime are not
sufficient to specify one.  To parse a time with a format that does not include
a time zone or UTC offset, you must specify the time zone explicitly.

    >>> parse_time("%DT%C", "2018-07-12T11:30:20", time_zone="America/New_York")
    ora.Time(2018, 7, 12, 15, 30, 20.00000000, UTC)


