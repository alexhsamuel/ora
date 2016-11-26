import bisect

from   .date import Date

#-------------------------------------------------------------------------------

# FIXME: Start, end dates.
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

    def __init__(self, min, max):
        self.__min = min
        self.__max = max


    def _check(self, date):
        date = Date(date)
        if not date.valid:
            raise InvalidDateError(date)  # FIXME: ?
        if date < self.__min:
            raise CalendarRangeError(
                "date {} before min {}".format(date, self.__min))
        if date > self.__max:
            raise CalendarRangeError(
                "date {} after max {}".foramt(date, self.__max))
        return date


    @property
    def min(self):
        return self.__min


    @property
    def max(self):
        return self.__max


    def last(self, date):
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
                date = self.last(date - 1)
        return date
        

            
class AllCalendar(Calendar):

    min = date.MIN
    max = date.MAX

    def __contains__(self, date):
        return True


    def last(self, date):
        return self._check(date)


    def next(self, date):
        return self._check(date)


    def shift(self, date, offset):
        return self._check(self._check(date) + offset)




class ExplicitCalendar(Calendar):

    def __init__(self, min, max, dates):
        super().__init__(min, max)
        self.__dates = sorted( self._check(d) for d in dates )


    def __contains__(self, date):
        date = self._check(date)
        i = bisect.bisect_left(self.__dates, date)
        return i != len(self.__dates) and self.__dates[i] == date


    def last(self, date):
        date = self._check(date)
        i = bisect.bisect_right(self.__dates, date)
        if i == 0:
            # FIXME
            pass
        else:
            return self.__dates[i]


    def next(self, date):
        date = self._check(date)
        i = bisect.bisect_left(self.__dates, date)
        if i == len(self.__dates):
            # FIXME
            pass
        else:
            return self.__dates[i]



        
        
