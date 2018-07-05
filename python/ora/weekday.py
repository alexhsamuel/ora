from   contextlib import suppress

import enum

#-------------------------------------------------------------------------------

class Weekday(enum.IntEnum):
    """
    A day of the (seven-day) week.

    Integer values are counted from Monday = 0.
    """

    Mon = 0
    Tue = 1
    Wed = 2
    Thu = 3
    Fri = 4
    Sat = 5
    Sun = 6

    def __repr__(self):
        return super().__str__()


    def __str__(self):
        return self.name



def to_weekday(obj):
    if isinstance(obj, Weekday):
        return obj
    with suppress(KeyError):
        return Weekday[str(obj)]
    with suppress(ValueError):
        return Weekday(int(obj))
    raise ValueError(f"not a weekday: {obj!r}")


# Add the days of the week to the module namespace.
globals().update(Weekday.__members__)


def weekday_range(weekday0, weekday1):
    """
    Returns weekdays between `weekday0` and `weekday1`, inclusive.

      >>> weekday_range(Mon, Fri)
      (Weekday.Mon, Weekday.Tue, Weekday.Wed, Weekday.Thu, Weekday.Fri)
      >>> weekday_range("Fri", "Mon")
      (Weekday.Fri, Weekday.Sat, Weekday.Sun, Weekday.Mon)

    """
    w0 = int(to_weekday(weekday0))
    w1 = int(to_weekday(weekday1))
    if w0 <= w1:
        res = range(w0, w1 + 1)
    else:
        res = ( w % 7 for w in range(w0, w1 + 8) )
    return tuple( Weekday(w) for w in res )


def parse_weekdays(string):
    """
    Parses a weekdays expression.

    The expression may consist of:
    - Empty, for no weekdays.
    - A single weekday abbreviation.
    - A range of weekdays, indicated with '-'.
    - A comma-separated list of the above.

        >>> parse_weekdays("")
        ()
        >>> parse_weekdays("Wed")
        (Weekday.Wed,)
        >>> parse_weekdays("Fri-Mon")
        (Weekday.Mon, Weekday.Fri, Weekday.Sat, Weekday.Sun)
        >>> parse_weekdays("Mon,Wed-Fri")
        (Weekday.Mon, Weekday.Wed, Weekday.Thu, Weekday.Fri)

    :return:
      A sorted sequence of weekdays.
    """
    weekdays = set()
    for part in str(string).split(","):
        if part == "":
            pass
        elif "-" in part:
            w0, w1 = part.split("-", 1)
            weekdays.update(weekday_range(w0, w1))
        else:
            weekdays.add(to_weekday(part))
    return tuple(sorted(weekdays))


