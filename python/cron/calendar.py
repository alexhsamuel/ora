import bisect

from   .date import Date

#-------------------------------------------------------------------------------

# FIXME: Start, end dates.

class Calendar:

    # FIXME: Exceptions?

    def __init__(self, min, max):
        self.__min = min
        self.__max = max


    @property
    def min(self):
        return self.__min


    @property
    def max(self):
        return self.__max


    def last(self, date):
        # FIXME: Check min.
        while date.valid and date not in self:
            date -= 1
        return date


    def next(self, date):
        # FIXME: Check max.
        while date.valid and date not in self:
            date += 1
        return date


    def shift(self, date, offset):
        # FIXME: Check min and max.
        assert date in self
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
        return date


    def next(self, date):
        return date


    def shift(self, date, offset):
        return date + offset




class SetCalendar(Calendar):

    def __init__(self, min, max, dates):
        super().__init__(min, max)
        dates = sorted( Date(d) for d in dates )
        # FIXME: Check valid.
        if len(dates) > 0:
            if dates[0] < self.min:
                raise ValueError("dates before min")
            if dates[-1] > self.max:
                raise ValueError("dates after max")
        self.__dates = dates;


    def __contains__(self, date):
        i = bisect.bisect_left(self.__dates, date)
        return i != len(self.__dates) and self.__dates[i] == date


    def last(self, date):
        i = bisect.bisect_right(self.__dates, date)
        if i == 0:
            # FIXME
            pass
        else:
            return self.__dates[i]


    def next(self, date):
        i = bisect.bisect_left(self.__dates, date)
        if i == len(self.__dates):
            # FIXME
            pass
        else:
            return self.__dates[i]



        
        
