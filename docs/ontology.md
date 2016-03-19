# Concepts

Time is a fundamentally straightforward concept.  Physical history is arranged into a linear sequence of instants.  We measure the span between two instances in this sequence with a unit of time, and identify any particular instant by the span from an arbitrary, fixed reference instant, much as we might identify stops along a straight road by distance markers placed along it.

Things get complicated when we start to measure time not in these terms, but in terms of repeating astronomical events, namely the full rotations of the earth (days) and its revolutions around the sun (years).  For historical reasons, we use a complex and arbitrary scheme for specifying times in these terms, which is not particularly amenable to representation in a computer.


## Time

A **time** is an infinitessimal instant in physical history.  A time
is independent of representation or time zone.  Two events are simultaneous if
and only if they occurred at the same time.

In principle, a time is infinitessimal; it has no duration.  In practice, a time is
represented by a finite-precision approximation, much as a real number is
approximated by a finite-precision floating point number.  A particular represented value of time corresponds to a (tiny) range of instantaneous times.

The standard unit for measuring time periods is the **second**.  As one second is a rather short unit of time by human perception, longer units are conventionally used: one **minute** is 60 seconds, and one **hour** is 60 minutes or 3,600 seconds.  (We don't, but might, use standard SI prefixes for larger units: the kilosecond, the megasecond, **etc.**  That's a rant for another day.)

The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) standard textual representation of a time looks like this:

    2016-03-15T09:09:17Z

This is actually the "extended" format, which includes extra punctuation to make it easier for humans to read.  The "basic" format for the same time is:

    20160315T090917Z

The representation consists of a date and daytime separated by 'T', and followed by a 'Z', which indicates that the date and daytime are interpreted in UTC time.  These are discussed below.  The last two digits, the "seconds" component of the daytime, may include a decimal point and fractional digits.


## Date

A **day** is a social construct that represents a period of time of usually
24 hours (86,400 seconds) corresponding roughly to one rotation of the Earth in a specific
geographical area.  A **date** specifies a particular day.

A date is specific to time zone geography; for example, Monday, 2016 March 15 in Tokyo represents a different 24-hour period than Monday, 2016 March 15 in New York.  That date in Tokyo occurs earlier but overlaps somewhat with the same date in New York.

A specific date in a specific time zone is generally, but not always, 24 hours
long.  For example, 2016 March 14 in the US/Eastern time zone is only 23 hours
long, as one hour is omitte to transition from standard to daylight
savings time.

There are many systems for specifiying date.  We consider here only the [proleptic Gregorian calendar](https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar), which is most widely used in the Western and business worlds.  

Dates are represented in a number of ways, using a complex numbering of time periods of historical and cultural import.  

  - A **year** is a range of either 365 or 366 days roughly corresponding to one revolution of the earth around the sun.  

  The pattern of year lengths is chosen to keep a year approximately synchronized with the actual orbit of the earth.  (There is no reason that the period earth's revolution should be an integral multiple of its orbital period, and in fact this is not the case.)  Approximately every fourth year is 366 days long; these are called **leap years**.

  Years are numbered consecutively forward from an arbitrarily chosen year 1 C.E.  (Years before the year 1 are counted consecutively backward from 1 B.C.E., skipping zero.  We don't consider B.C.E. years here, and omit the C.E. qualification for years going forward.)

  - Years are further partitioned into 12 **months**, arbitrary date intervals of 28, 29, 30, or 31 days.  The 12 months have names in each human language (in English, "January", "February", ...), and are numbered conventionally 1 (_not zero!_) through 12.

  Each month has a fixed number of days, either 30 or 31, except the second month, February, which has 28 days in ordinary years, 29 days in leap years.  

  - Further, days are partitioned into seven-day periods called **weeks**.  Weeks are always exactly seven days long and follow consecutively; they are not synchronized to months or years.  The seven **weekdays** (days of the week) have traditional names in each human language (in English, "Sunday", "Monday", ...).  

  There is no universally accepted convention for which weekday starts the week, which causes some confusion.  For example, in the United States, the week is usually considered to start on Sunday, while in many other countries, the week is considered to start on Monday.

  The seven weekdays are also numbered.  Numberings 0 through 6 and 1 through 7 are both used, and the lowest-numbered weekday is not standard.  Check numbering conventions carefully when using weekday numbers.

Given these (somewhat arbitrary) definitions, a number of textual representations for dates are used.

  - In the most common representation, a date is given by a triple:

    - a year number,
    - a month (either by name or number), and
    - a day number within the month counting from 1 (_not zero!_).  

  This is the default representation for a date, unless otherwise indicated.  The ISO 8601 textual representation looks like this (note the zero padding, so that the representation is always 10 characters long):

    ```
    2016-03-16
    ```

    Many other textual representations are [commonly used](https://xkcd.com/1179/) in practice.

  - In the [**week date**](https://en.wikipedia.org/wiki/ISO_week_date) representation, the date is given by a year number, a week number in that year counting from 1 (_not zero!_), and a weekday.  _The year number in the week date is different than the year number in the common date representation,_ such that each numbered week falls entirely within a one year or another.  (Remember that weeks are not synchronized to years.)   

  The ISO 8601 textual representation of a week date looks like this:

  ```
  2016-W11-2
  ```

  The weekday is counted 1 through 7, starting on Monday.

  - In the **ordinal date** representation, the date is specified as the same year number as the common date representation, but days within the year are numbered sequentially 1 (_not zero!_) through 365 or 366, rather than divided into months.

  The ISO 8601 textual representation of an ordinal date looks like this (note the zero padding, so that the representation is always 8 characters long):

  ```
  2016-076
  ```

The default year, month, day components are often encoded in an eight-digit decimal integer, instead of a string, for example 20160316, called the **YMDI** (year-month-day integer) representation.  This representation is discouraged, but supported by Cron.


## Daytime

A **daytime** (time of day) represents one of the instants within a given date.  Like a time, a daytime is infinitessimal, and is in practice is represented by a
finite-precision approximation.

Daytimes are measured forward from **midnight**, the first instant of any date.  Daytime can be represented most easily as seconds since midnight (**SSM**).  

Conventionally, however, daytime is represented in hours, minutes, and seconds.  Hours are numbered as integers 0 through 23; minutes as integers from 0 through 59; and seconds as real numbers in the half-open range [0, 60).  

The ISO 8601 textual representation of a daytime looks like this (note the zero padding):

```
09:09:17
```

As with times, the last component (seconds) may include a decimal point and fractional digits.  


## Local Time and UTC

Conventionally, different geographical regions of the world set their clocks and calendars differently.  The clocks (usually) tick forward at the same rate, so that between any two geographical points, there is a fixed time offset between the two dates and daytimes.

Instead of tracking all pairwise offsets, we can instead specify the offset between any geographical region and an arbitrary standard local time.  This standard local time is called **[UTC](https://en.wikipedia.org/wiki/Coordinated_Universal_Time)** (for Coordinated Universal Time).

A (date, daytime) pair is a **local time**, and is meaningful only within a certain geographical region.


## Time Zones

A **time zone** is a sociopolitical designation of a geographic area that fixes
the mapping of times to local times.  A time zone specifies what all calendars and clocks within its borders should read at any specific physical time.  In effect, time zones together specify a function _to&#95;local_:

> _to&#95;local_ : time, time zone &rarr; date, daytime

A time zone maps time to local time by specifying, for any given time, a **UTC offset**, which is a positive or negative time shift between -12 hours and + 12 hours (but isn't necessarily a round number of hours).  Thus, _to&#95;local_ is equivalently given by another function _offset_:

> _offset_ : time, time zone &rarr; UTC offset

Note that in English, "time zone" is two words; so, not <del>timezone</del>.   


## Localizing time

Given a time and a time zone, the local time is obtained as follows:

1. Look at the UTC calendar / clock at that time, and read off its date and daytime.
1. Find the UTC offset corresponding to that time zone and time.
1. Adjust the UTC daytime by the UTC offset, shifting the UTC date forward or backward if the daytime rolls backward or forward a day.

For example, at time 2016-03-19T01:18:40Z, the offset in the US/Eastern time zone (a.k.a. America/New_York) is -14,400 seconds (_i.e._ 4 hours behind UTC).  The local time comprises the date 2016-03-18 and daytime 21:18:40.

Time zones are defined by statute as matters of social policy. UTC, however, is not a political designation, and serves as a universal reference. As such, times are represented most straightforwardly as dates and daytimes in UTC.  


## Daylight Saving Time

A time zone's UTC offset is not necessarily constant over time. Some time zones institute **daylight saving time** (DST) (or "summer time" in some jurisdictions), during which the mapping is abruptly shifted forward or backward, in an attempt to encourage particular social and economic behavior. This is the reason that _to&#95;local_ and _offset_ are both functions of both time zone and time.


## Time Zones _vs_ Offsets

Note that a UTC offset (often appended as a +HH:MM suffix to rendered times), does _not_ specify a time zone.  A UTC offset may be specified by multiple time zones at a given time, and a time zone's UTC offset can be different for different times.

For example, the US/Eastern time zone specifies a UTC offset of -5 hours during the winter months (during which it is known as "Eastern Standard Time", or "EST") but -4 hours during the summer months (when it is known as "Eastern Daylight Time", or "EDT"). As such, EST and EDT are not time zones! They are monickers for the UTC offset used for part of the year in the Eastern United States time zone.  

Similarly, UTC-5 isn't a really time zone! It's a UTC offset that happens to be used in the Eastern United States in the winter months. But it also happens to be used in the Central United States during the summer months, and also in a number of countries and parts of countries, for example Colombia, year around.

We can nevertheless imagine a time zone that is always five hours behind UTC, and always will be, regardless of the opinions of any particular government.  This is what the UTC-5 "time zone" would be&mdash;however, it is the time zone for no physical part of the earth.

We can also, for convenience, think of UTC itself as a time zone whose UTC offset is always zero.  It too is not honored anywhere on earth. (The time zone comprising Iceland happens to have a UTC offset of zero all year, but the government of Iceland is free to change this at its whim, whereas UTC will not change by definition.)


## Localization

**Localization** maps a time to a date and daytime in a given time zone.  The
localization function, for valid dates and daytimes,

> time → (date, daytime, time zone)

is approximately bijective.  The bijection is
violated only at transitions to and from DST, near which a given (date, daytime,
time zone) triple may correspond to either zero or two times.
