from   ora import *

#-------------------------------------------------------------------------------

TEST_DATE_PARTS = (
    (   1, Jan,  1),
    (   1, Jan,  2),
    (   1, Feb,  1),
    (   1, Feb, 28),
    (   1, Mar,  1),
    (   1, Dec, 31),
    (   2, Jan,  1),
    (   3, Jan,  1),
    (   3, Dec, 31),
    (   4, Jan,  1),
    (   4, Feb, 28),
    (   4, Feb, 29),
    (   4, Mar,  1),
    (  99, Dec, 31),
    ( 100, Jan,  1),
    ( 100, Feb, 28),
    ( 100, Mar,  1),
    ( 101, Jan,  1),
    ( 400, Jan,  1),
    ( 400, Feb, 28),
    ( 400, Feb, 29),
    ( 400, Mar,  1),
    ( 401, Jan,  1),
    (1066, Oct, 14),
    (1944, Jun,  6),
    (1973, Dec,  3),
    (1999, Dec, 31),
    (2000, Jan,  1),
    (9999, Jan,  1),
    (9999, Dec, 31),
    )


#-------------------------------------------------------------------------------

def sample_dates(interval=137, Date=Date):
    yield Date.MIN
    for datenum in range(Date.MIN.datenum + 1, Date.MAX.datenum + 1, interval):
        yield Date.from_datenum(datenum)


