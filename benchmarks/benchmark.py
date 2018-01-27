from   contextlib import suppress
from   time import perf_counter
import itertools
import logging
import numpy as np
import pytz

logging.basicConfig(level=logging.INFO)


class timing:
    """
    Context manager that measures wall clock time between entry and exit.
    """

    def __init__(self, name, timer=perf_counter):
        self.__name = name
        self.__timer = timer
        self.__start = self.__end = self.__elapsed = None
        self.__end = None


    def __enter__(self):
        self.__start = self.__timer()
        return self
        

    def __exit__(self, *exc):
        self.__end = self.__timer()


    @property
    def name(self):
        return self.__name


    @property
    def start(self):
        return self.__start


    @property
    def end(self):
        return self.__end


    @property
    def elapsed(self):
        return self.__end - self.__start



def _benchmark(fn, s, n, *, quantile=0.05):
    # Loop pedestal calculation.
    null = lambda: None
    samples = []
    for _ in range(s):
        start = perf_counter()
        for _ in range(n):
            null()
        samples.append(perf_counter() - start)
    pedestal = np.percentile(samples, 100 * quantile, interpolation="nearest")

    logging.debug("pedestal={:.0f} ns".format(pedestal / n / 1e-9))
    
    samples = []
    for _ in range(s):
        start = perf_counter()
        for _ in range(n):
            fn()
        samples.append(perf_counter() - start)
    
    result = np.percentile(samples, 100 * quantile, interpolation="nearest")
    return (result - pedestal) / n


def benchmark(fn, *, quantile=0.05):
    MIN_SAMPLE_TIME = 1E-3
    TARGET_TIME = 0.2

    # Estimate parameters.
    for scale in itertools.count():
        n = 10 ** scale
        start = perf_counter()
        for _ in range(n):
            fn()
        elapsed = perf_counter() - start
        if elapsed >= MIN_SAMPLE_TIME:
            break

    s = max(5, min(100, int(TARGET_TIME / elapsed)))

    logging.debug("calibration: n={} s={}".format(n, s))

    return _benchmark(fn, s, n, quantile=quantile)


#-------------------------------------------------------------------------------

# FIXME: Things to benchmark:
# - to_local
#   - @
#   - UTC vs other zones
# - from_local
# - to/from epoch s, ns
# - conversion among subtypes
# - convert from one tz to another
# - parse

def benchmark_raw_now():
    from time import time
    yield "time.time()"             , benchmark(lambda: time())

    import datetime
    utcnow = datetime.datetime.utcnow
    yield "datetime.utcnow()"       , benchmark(lambda: utcnow())

    from ora import now, Time, SmallTime, NsecTime
    yield "ora.now()"               , benchmark(lambda: now())
    yield "ora.now(Time)"           , benchmark(lambda: now(Time))
    yield "ora.now(SmallTime)"      , benchmark(lambda: now(SmallTime))
    yield "ora.now(NsecTime)"       , benchmark(lambda: now(NsecTime))

    with suppress(ImportError):
        from ora import NsTime
        yield "ora.now(NsTime)"     , benchmark(lambda: now(NsTime))


def benchmark_utc_now():
    import datetime
    now = datetime.datetime.now
    utcnow = datetime.datetime.utcnow
    from pytz import UTC
    yield "datetime.utcnow()"       , benchmark(lambda: utcnow())
    yield "datetime.now(UTC)"       , benchmark(lambda: now(UTC))

    from ora import now, Time, SmallTime, NsecTime, UTC, to_local
    yield "ora.now() @ UTC"         , benchmark(lambda: now() @ UTC)
    yield "to_local(ora.now(), UTC)", benchmark(lambda: to_local(now(), UTC))
    yield "ora.now(Time) @ UTC"     , benchmark(lambda: now(Time) @ UTC)
    yield "ora.now(SmallTime) @ UTC", benchmark(lambda: now(SmallTime) @ UTC)
    yield "ora.now(NsecTIme) @ UTC" , benchmark(lambda: now(NsecTime) @ UTC)

    with suppress(ImportError):
        from ora import NsTime
        yield "ora.now(NsTime) @ UTC", benchmark(lambda: now(NsTime) @ UTC)


def benchmark_local_now():
    import datetime
    now = datetime.datetime.now
    yield "datetime.now()"          , benchmark(lambda: now())

    from ora import now, Time, SmallTime, NsecTime, get_display_time_zone, to_local
    z = get_display_time_zone()
    yield "ora.now() @ z"           , benchmark(lambda: now() @ z)
    yield "to_local(ora.now(), z)",   benchmark(lambda: to_local(now(), z))
    yield "ora.now(Time) @ z"       , benchmark(lambda: now(Time) @ z)
    yield "ora.now(SmallTime) @ z"  , benchmark(lambda: now(SmallTime) @ z)
    yield "ora.now(NsecTIme) @ z"   , benchmark(lambda: now(NsecTime) @ z)

    with suppress(ImportError):
        from ora import NsTime
        yield "ora.now(NsTime) @ z" , benchmark(lambda: now(NsTime) @ z)


def benchmark_tz_now():
    import datetime
    now = datetime.datetime.now
    z = pytz.timezone("America/New_York")
    yield "datetime.now(z)"         , benchmark(lambda: now(z))

    from ora import now, Time, SmallTime, NsecTime, TimeZone, to_local
    z = TimeZone("America/New_York")
    yield "ora.now() @ z"           , benchmark(lambda: now() @ z)
    yield "to_local(ora.now(), z)"  , benchmark(lambda: to_local(now(), z))
    yield "ora.now(Time) @ z"       , benchmark(lambda: now(Time) @ z)
    yield "ora.now(SmallTime) @ z"  , benchmark(lambda: now(SmallTime) @ z)
    yield "ora.now(NsecTime) @ z"   , benchmark(lambda: now(NsecTime) @ z)

    with suppress(ImportError):
        from ora import NsTime
        yield "ora.now(NsTime) @ z" , benchmark(lambda: now(NsTime) @ z)


def benchmark_time_literal():
    from datetime import datetime
    yield "datetime(...)"           , benchmark(lambda: datetime(2018, 1, 6, 16, 51, 45, 123456))

    import pytz
    tz = pytz.timezone("America/New_York")
    yield "tz.localize(datetime(...))", benchmark(lambda: tz.localize(datetime(2018, 1, 6, 16, 51, 45, 123456)))

    import dateutil.tz
    tz = dateutil.tz.gettz("America/New_York")
    yield "datetime(..., tz)", benchmark(lambda: datetime(2018, 1, 6, 16, 51, 45, 123456, tz))

    from ora import Date, Daytime, Time, TimeZone, from_local, Jan
    tz = TimeZone("America/New_York")
    yield "(Date(...), Daytime(...)) @ tz", benchmark(lambda: (Date(2018, 1, 6), Daytime(16, 51, 45.123456)) @ tz)
    yield "(Y/M/D, Daytime(...)) @ tz", benchmark(lambda: (2018/Jan/6, Daytime(16, 51, 45.123456)) @ tz)
    yield "from_local((Date(...), Time(...)), tz)", benchmark(lambda: from_local((Date(2018, 1, 6), Daytime(16, 51, 45.123456)), tz))
    yield "from_local((Date(...), Time(...)), 'tz')", benchmark(lambda: from_local((Date(2018, 1, 6), Daytime(16, 51, 45.123456)), "America/New_York"))
    yield "Time(..., tz)"           , benchmark(lambda: Time(2018, 1, 6, 16, 51, 45.123456, tz))
    yield "Time(..., 'tz')"         , benchmark(lambda: Time(2018, 1, 6, 16, 51, 45.123456, "America/New_York"))


def benchmark_convert_tz():
    import datetime
    import pytz
    tz0 = pytz.timezone("America/New_York")
    tz1 = pytz.timezone("Asia/Tokyo")
    t = datetime.datetime(2018, 1, 5, 21, 17, 56, 123456)
    yield "localize().astimezone()" , benchmark(lambda: tz0.localize(t).astimezone(tz1))

    import dateutil.tz
    tz0 = dateutil.tz.gettz("America/New_York")
    tz1 = dateutil.tz.gettz("Asia/Tokyo")
    t = datetime.datetime(2018, 1, 5, 21, 17, 56, 123456)
    yield "replace(tzinfo).astimezone()", benchmark(lambda: t.replace(tzinfo=tz0).astimezone(tz1))

    t = datetime.datetime(2018, 1, 5, 21, 17, 56, 123456, tz0)
    yield "t.astimezone()"          , benchmark(lambda: t.astimezone(tz1))

    from ora import Date, Daytime, Time, TimeZone, to_local, from_local
    tz0 = TimeZone("America/New_York")
    tz1 = TimeZone("Asia/Tokyo")
    t = Date(2018, 1, 5), Daytime(21, 17, 56.123456)
    yield "t @ tz0 @ tz1"            , benchmark(lambda: t @ tz0 @ tz1)
    yield "to_local(from_local(t, tz0), tz1)", benchmark(lambda: to_local(from_local(t, tz0), tz1))
    with suppress(ImportError):
        from ora import NsDaytime
        t = Date(2018, 1, 5), NsDaytime(21, 17, 56.123456)
        yield "t @ tz0 @ tz1 [NsDaytime]", benchmark(lambda: t @ tz0 @ tz1)
        yield "to_local(from_local(t, tz0), tz1)", benchmark(lambda: to_local(from_local(t, tz0), tz1))


def benchmark_today_local():
    import datetime
    today = datetime.date.today
    yield "date.today()"            , benchmark(lambda: today())

    from ora import today, get_display_time_zone, Date, Date16
    z = get_display_time_zone()
    yield "ora.today(z)"            , benchmark(lambda: today(z))
    yield "ora.today('display')"    , benchmark(lambda: today("display"))
    yield "ora.today(z, Date)"      , benchmark(lambda: today(z, Date))
    yield "ora.today('display', Date)", benchmark(lambda: today("display", Date))
    yield "ora.today(z, Date16)"    , benchmark(lambda: today(z, Date16))
    yield "ora.today('display', Date16)", benchmark(lambda: today("display", Date16))


def benchmark_time_format():
    import datetime
    t = datetime.datetime.utcnow()
    yield "str(datetime)"           , benchmark(lambda: str(t))
    yield "format(datetime, py_fmt)", benchmark(lambda: format(t, "%Y-%m-%d %H:%M:%S.%f"))
    yield "format(datetime, rfc)"   , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%S.%fZ"))

    import ora
    t = ora.now()
    yield "str(Time)"               , benchmark(lambda: str(t))
    yield "format(Time, py_fmt)"    , benchmark(lambda: format(t, "%Y-%m-%d %H:%M:%.6S"))
    yield "format(Time, rfc)"       , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%.6SZ"))
    yield "format(Time, dstrfc)"    , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%.6SZ@"))
    yield "format(Time, tzrfc)"     , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%.6SZ@America/New_York"))
    yield "format(Time, utcrfc)"    , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%.6SZ@UTC"))
    yield "Time.format(rfc)"        , benchmark(lambda: t.format("%Y-%m-%d"))
    yield "Time.format(rfc, tzname)", benchmark(lambda: t.format("%Y-%m-%d", "America/New_York"))
    z = ora.TimeZone("America/New_York")
    yield "Time.format(rfc, z)"     , benchmark(lambda: t.format("%Y-%m-%d", z))


def benchmark_date_format():
    import datetime
    d = datetime.date(2018, 1, 14)
    yield "str(date)"               , benchmark(lambda: str(d))
    yield "format(date, fmt)"       , benchmark(lambda: format(d, "%Y-%m-%d"))

    import ora
    d = ora.Date(2018, 1, 14)
    yield "str(Date)"               , benchmark(lambda: str(d))
    yield "format(Date, fmt)"       , benchmark(lambda: format(d, "%Y-%m-%d"))
    yield "format(Date, tzfmt)"     , benchmark(lambda: format(d, "%Y-%m-%d@America/New_York"))


def benchmark_time_comparison():
    import datetime
    t0 = datetime.datetime.utcnow()
    t1 = datetime.datetime.utcnow()
    yield "datetime =="             , benchmark(lambda: t0 == t1)
    yield "datetime !="             , benchmark(lambda: t0 != t1)
    yield "datetime <"              , benchmark(lambda: t0 <  t1)
    yield "datetime <="             , benchmark(lambda: t0 <= t1)

    import ora
    t0 = ora.now()
    t1 = ora.now()
    yield "Time =="                 , benchmark(lambda: t0 == t1)
    yield "Time !="                 , benchmark(lambda: t0 != t1)
    yield "Time < "                 , benchmark(lambda: t0 <  t1)
    yield "Time <="                 , benchmark(lambda: t0 <= t1)


def summarize(benchmarks):
    for name, elapsed in benchmarks:
        print("{:40s} {:7.3f} µs".format(name, elapsed / 1e-6))
    print()


summarize(benchmark_raw_now())
summarize(benchmark_utc_now())
summarize(benchmark_local_now())
summarize(benchmark_tz_now())
summarize(benchmark_time_literal())
summarize(benchmark_convert_tz())
summarize(benchmark_today_local())
summarize(benchmark_time_format())
summarize(benchmark_date_format())
summarize(benchmark_time_comparison())

