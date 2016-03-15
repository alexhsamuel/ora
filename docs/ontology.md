Time
====

In principle, a _time_ is an infinitessimal instant in physical history.  A time
is independent of representation or time zone.  Two events are simultaneous if
and only if they occurred at the same time.

A time is infinitessimal; it has no duration.  In practice, a time is
represented by a finite-precision approximation, much as a real number is
approximated by a finite-precision floating point number.


Date
====

A _date_ is a social construct that represents a period of time of approximately
24 hours corresponding roughly to one rotation of the Earth in a specific
geographical area.

A date is specific to time zone geography; for example, Monday, 2016 March 15 in
Tokyo starts and ends earlier than Monday, 2016 March 15 in New York.  The same
date in these two time zones overlaps only for a few hours.

A specific date in a specific time zone is generally, but not always, 24 hours
long.  For example, 2016 March 14 in the US/Eastern time zone is only 23 hours
long, as one hour was lost due to the transition from standard to daylight
savings time.


Daytime
=======

A _daytime_ (time of day) represents one of the instants within a given date.
For example, "noon" is a particular instant within some (unspecified) date.

Like a time, a daytime is infinitessimal, and is in practice is represented by a
finite-precision approximation.


Time Zone
=========

A _time zone_ is a sociopolitical designation of a geographic area in which
the mapping of times to dates and daytimes is uniform over the area.  The set
of time zones is a partition of the surface of the earth.

For a given time zone, the mapping from times to dates and daytimes is not
necessarily smooth and linear.  Some time zones institute _daylight savings
time_ (DST) (or "summer time" in some jurisdictions), during which the mapping
is abruptly shifted by one hour, in an attempt to encourage particular social
and economic behavior.


Conversions
===========

_Localization_ maps a time to a date and daytime in a given time zone.  The
mapping,

    time <-> (date, daytime, time_zone)

for valid dates and daytimes is approximately bijective.  The bijection is
violated only at transitions to and from DST, near which a given (date, daytime,
time_zone) triple may correspond to either zero or two times.

