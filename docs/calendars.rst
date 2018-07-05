Calendars
=========

A calendar is a subset of the dates in a date range.  It may represent, for 
instance, public holidays in a jurisdiction, or dates on which a business is
open.  

A calendar always carries a range of dates for which it is valid.  Calendar
queries outside of this date range are invalid, and raise `CalendarRangeError`.
Dates outside the range are not implicitly in or not in the calendar.

`Calendar` represents a calendar.  To create one, specify the range as a
`(start, stop)` pair, and an iterable of the dates that are in the calendar.
The range is half-inclusive; dates on or after `start` and strictly before
`stop` are in the range.

     >>> cal_range = Date(2018, 1, 1), Date(2019, 1, 1)
     >>> holidays = Calendar(cal_range, (
     ...     Date(2018,  1,  1),  # New Years Day
     ...     Date(2018,  1, 15),  # Martin Luther King Day
     ...     Date(2018,  5, 28),  # Memorial Day
     ...     Date(2018,  7,  4),  # Independence Day
     ...     Date(2018,  9,  3),  # Labor Day
     ...     Date(2018, 11, 22),  # Thanksgiving Day
     ...     Date(2018, 12, 25),  # Christmas Day
     ... ))

All the dates in the calendar must be in the range.

Optionally, the calendar can carry a name; use the `name` attribute (initially
`None`).

Membership
----------

Use the `in` operator or `contains` method to test membership, for a date or any
value that can be converted to a date.

    >>> Date(2018, 1, 15) in holidays
    True
    >>> holidays.contains("2018-01-15")
    True

Remember, the calendar can only test membership for dates in its range.

    >>> holidays.range
    (Date(2018, Jan, 1), Date(2019, Jan, 1))
    >>> Date(2017, 12, 31) in holidays
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ora.CalendarRangeError: date not in calendar range

Shifts
------

The `before` method returns the latest calendar date on or before a date.

    >>> holidays.before(Date(2018, 2, 1))
    Date(2018, Jan, 15)

The result is that date itself, if it is in the calendar.

    >>> holidays.before(Date(2018, 1, 15))
    Date(2018, Jan, 15)

Likewise, `after` returns the earliest calendar date on or after a date.

    >>> holidays.after(Date(2018, 2, 1))
    Date(2018, May, 28)

Given a date in the calendar, to find the previous or next calendar date,
subtract or add one first.

    >>> mlk = Date(2018, 1, 15)
    >>> holidays.after(mlk + 1)
    Date(2018, May, 28)

To shift multiple calendar days forward, use the `shift` method.  Use a
positiveargument to shift forward, negative for backward.

    >>> holidays.shift(mlk, 3)
    Date(2018, Sep, 3)

If these methods move past the calendar range, the calendar throws
`CalendarRangeError`.

    >>> holidays.shift(mlk, -3)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ora.CalendarRangeError: date not in calendar range

Making calendars
----------------

In addition specifying to explicit dates, you can create calendars with
these special functions:

`make_const_calendar(range, contains)` returns a calendar that contains *all*
dates in its range, if `contains` is true, or *none* of them otherwise.

`make_weekday_calendar(range, weekdays)` returns a calendar that contains only
the specified weekdays.

    >>> cal = make_weekday_calendar(cal_range, [Mon, Wed, Fri])

Loading calendars
-----------------

The `load_calendar_file` function takes a path and loads a calendar from a text
file in this format:

::

    START 2010-01-01 
    STOP 2021-01-01
    2010-01-01
    2010-01-18
    2010-02-15

Blank lines are removed; as is text following each date, which may be used for
comments.

::

    START 2010-01-01 
    STOP 2021-01-01

    2010-01-01 New Year's Day
    2010-01-18 Birthday of Martin Luther King, Jr.
    2010-02-15 Washington's Birthday

Use `parse_calendar` to parse lines of text directly.

Dumping calendars
-----------------

Use `format_calendar` to produce the calendar file format.  This function
returns an interable of lines.

    >>> for line in format_calendar(cal):
    ...     print(line)

To write this directly to a file, use `dump_calenar_file(cal, path)`.    


Arithmetic
----------

A calendar is, in a sense, a boolean mask over the dates in its range.
Calendars can be combined using bitwide arithmetic.

The `~` operator returns an inverted calendar, with dates in the range *not* in
the original calendar.

The `&`, `|`, and `^` operators take two calendars, and return the intersection,
union, and symmetric difference, respectively.  The range of the combined
calendar is always the intersection (overlap) of the two ranges.

    >>> week_cal = make_weekday_calendar(cal_range, [Mon, Tue, Wed, Thu, Fri])
    >>> work_cal = week_cal & ~holidays

