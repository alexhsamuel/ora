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
    
The representation consists of a date and daytime separated by 'T', and followed by a 'Z', which indicates that the date and daytime are interpreted in the UTC time zone.  These are discussed below.  The last two digits, the "seconds" component of the daytime, may include a decimal point and fractional digits.


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

## Daytime

A **daytime** (time of day) represents one of the instants within a given date.  Like a time, a daytime is infinitessimal, and is in practice is represented by a
finite-precision approximation.

Daytimes are measured forward from **midnight**, the first instant of any date.  Daytime can be represented most easily as seconds since midnight (**SSM**).  

Conventionally, however, daytime is represented in hours, minutes, and seconds.  Hours are numbered as integers 0 through 23, or 22 or 24 on DST transition dates; minutes as integers from 0 through 59; and seconds as real numbers in the half-open range [0, 60).  

The ISO 8601 textual representation of a daytime looks like this (note the zero padding):

```
09:09:17
```

As with times, the last component (seconds) may include a decimal point and fractional digits.  


## Time Zone

A **time zone** is a sociopolitical designation of a geographic area in which
the mapping of times to dates and daytimes is uniform over the area.  The set
of time zones is a partition of the surface of the earth.

For a given time zone, the mapping from times to dates and daytimes is not
necessarily smooth and linear.  Some time zones institute **daylight savings
time** (DST) (or "summer time" in some jurisdictions), during which the mapping
is abruptly shifted forward or backward, in an attempt to encourage particular social and economic behavior.

Most time zones are defined by statute as matters of social policy. [**UTC** ](https://en.wikipedia.org/wiki/Coordinated_Universal_Time), however, is not a political designation, uses no DST, and serves as a universal reference.  As such, times are represented most straightforwardly as dates and daytimes in UTC.  

Each time zone, _at any given time_, specifies a time offset from UTC.  This offset is used to adjust that particular time when representing it as date and daytime in that time zone.  Note that an offset from UTC is not a time zone and does not specify a time zone; nor does a time zone specify a unique offset from UTC.  Because of DST, a time zone may specify different offsets from UTC during different parts of the year.  For example, the time zone US/Eastern is five hours behind UTC during the winter months (known as Eastern Standard Time, or EST) but four hours behind UTC during the summer months (known as Eastern Daylight Time, or EDT).  There are various other time zones that specify the same offsets from UTC during various parts of the year.

Note that in English, "time zone" is two words; so, not <del>timezone</del>.  Also note that UTC is _not_ technically the same as Greenwich Mean Time (GMT); GMT is used during winter months by the U.K. and happens to have a zero offset from UTC.  



## Localization

**Localization** maps a time to a date and daytime in a given time zone.  The
localization function, for valid dates and daytimes,

> time → (date, daytime, time zone)

is approximately bijective.  The bijection is
violated only at transitions to and from DST, near which a given (date, daytime,
time zone) triple may correspond to either zero or two times.

