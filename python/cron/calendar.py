import bisect
from   collections import namedtuple

from   . import Weekday
from   .date import Date

#-------------------------------------------------------------------------------

# FIXME: Elsewhere.
Range = namedtuple("Range", ("min", "max"))


# FIXME: What about invalid and missing?

class CalendarError(Exception):

    pass


class CalendarRangeError(CalendarError):
    """
    A date is not in the range of the calendar.
    """

    pass



class DateNotInCalendarError(CalendarError):

    pass



class Calendar:

    # FIXME: Exceptions?

    def __init__(self, range):
        self.__range = Range(*range)


    def _check(self, date):
        date = Date(date)
        if not date.valid:
            raise InvalidDateError(date)  # FIXME: ?
        if date < self.__range.min:
            raise CalendarRangeError(
                "date {} before min {}".format(date, self.__range.min))
        if date > self.__range.max:
            raise CalendarRangeError(
                "date {} after max {}".foramt(date, self.__range.max))
        return date


    @property
    def range(self):
        """
        The range of dates covered by this calendar.
        """
        return self.__range


    def previous(self, date):
        date = self._check(date)
        while date.valid and date not in self:
            date -= 1
        return date


    def next(self, date):
        date = self._check(date)
        while date.valid and date not in self:
            date += 1
        return date


    def shift(self, date, offset):
        date = self._check(date)
        if date not in self:
            raise DateNotInCalendarError(date)
        if offset > 0:
            for _ in range(offset):
                date = self.next(date + 1)
        elif offset < 0:
            for _ in range(offset):
                date = self.previous(date - 1)
        return date
        


class AllCalendar(Calendar):

    range = Range(Date.MIN, Date.MAX)

    def __contains__(self, date):
        return True


    def previous(self, date):
        return self._check(date)


    def next(self, date):
        return self._check(date)


    def shift(self, date, offset):
        return self._check(self._check(date) + offset)




class ExplicitCalendar(Calendar):

    def __init__(self, range, dates):
        super().__init__(range)
        self.__dates = sorted( self._check(d) for d in dates )


    def __contains__(self, date):
        date = self._check(date)
        i = bisect.bisect_left(self.__dates, date)
        return i != len(self.__dates) and self.__dates[i] == date


    def previous(self, date):
        date = self._check(date)
        i = bisect.bisect_right(self.__dates, date)
        if i == 0:
            raise CalendarRangeError(
                "no calendar previous date for {}".format(date))
        else:
            return self.__dates[i]


    def next(self, date):
        date = self._check(date)
        i = bisect.bisect_left(self.__dates, date)
        if i == len(self.__dates):
            raise CalendarRangeError(
                "no calendar next date for [{}]".format(date))
        else:
            return self.__dates[i]

        
        
class WeekdayCalendar(Calendar):

    def __init__(self, weekdays):
        super().__init__(Range(Date.MIN, Date.MAX))
        self.__weekdays = { Weekday(w) for w in weekdays }


    def __contains__(self, date):
        return date.weekday in self.__weekdays


    # FIXME: We can make previous() and next() more efficient.



def parse_calendar(lines):
    # FIXME: Min and max!

    # Remove whitespace and trailing comments.
    lines = ( l.split("#", 1)[0].strip() for l in lines )
    # Skip blank lines.
    lines = ( l for l in lines if l != "" )
    # FIXME: Handle errors better.
    dates = sorted( Date(l) for l in lines )
    assert len(dates) > 0  # FIXME

    return ExplicitCalendar(Range(dates[0], dates[-1] + 1), dates)


def load_calendar_file(path):
    from pathlib import Path  # FIXME
    with Path(path).open() as file:
        return parse_calendar(file)


if __name__ == "__main__":
    import cron
    # cal = WeekdayCalendar({cron.Mon, cron.Tue, cron.Wed, cron.Thu, cron.Fri})
    cal = load_calendar_file("share/calendar/US federal holidays.txt")
    date = cron.today("UTC")
    print(date)
    date = cal.next(date)
    while True:
        print(date, date.weekday)
        date = cal.shift(date, 1)


