from   argparse import ArgumentParser
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
    yield "time"    , "time.time()"     , benchmark(lambda: time())

    import datetime
    utcnow = datetime.datetime.utcnow
    yield "datetime", "utcnow()"        , benchmark(lambda: utcnow())

    from ora import now, Time, SmallTime, NsTime, Time128
    yield "ora", "now()"                , benchmark(lambda: now())
    yield "ora", "now(Time)"            , benchmark(lambda: now(Time))
    yield "ora", "now(SmallTime)"       , benchmark(lambda: now(SmallTime))
    yield "ora", "now(NsTime)"          , benchmark(lambda: now(NsTime))
    yield "ora", "now(Time128)"         , benchmark(lambda: now(Time128))


def benchmark_utc_now():
    import datetime
    now = datetime.datetime.now
    utcnow = datetime.datetime.utcnow
    UTC = datetime.timezone.utc
    yield "datetime", "utcnow()"        , benchmark(lambda: utcnow())
    yield "datetime", "now(UTC)"        , benchmark(lambda: now(UTC))

    from ora import now, Time, SmallTime, NsTime, Time128, UTC, to_local
    yield "ora", "now() @ UTC"          , benchmark(lambda: now() @ UTC)
    yield "ora", "to_local(now(), UTC)" , benchmark(lambda: to_local(now(), UTC))
    yield "ora", "now(Time) @ UTC"      , benchmark(lambda: now(Time) @ UTC)
    yield "ora", "now(SmallTime) @ UTC" , benchmark(lambda: now(SmallTime) @ UTC)
    yield "ora", "now(NsTime) @ UTC"    , benchmark(lambda: now(NsTime) @ UTC)
    yield "ora", "now(Time128) @ UTC"   , benchmark(lambda: now(Time128) @ UTC)


def benchmark_local_now():
    import datetime
    now = datetime.datetime.now
    yield "datetime", "now()"           , benchmark(lambda: now())

    from ora import now, Time, SmallTime, NsTime, Time128, get_display_time_zone, to_local
    z = get_display_time_zone()
    yield "ora", "now() @ z"            , benchmark(lambda: now() @ z)
    yield "ora", "to_local(now(), z)"   , benchmark(lambda: to_local(now(), z))
    yield "ora", "now(Time) @ z"        , benchmark(lambda: now(Time) @ z)
    yield "ora", "now(SmallTime) @ z"   , benchmark(lambda: now(SmallTime) @ z)
    yield "ora", "now(NsTime) @ z"      , benchmark(lambda: now(NsTime) @ z)
    yield "ora", "now(Time128) @ z"     , benchmark(lambda: now(Time128) @ z)


def benchmark_tz_now():
    import datetime
    now = datetime.datetime.now
    z = pytz.timezone("America/New_York")
    yield "datetime", "now(z)"          , benchmark(lambda: now(z))

    from ora import now, Time, SmallTime, NsTime, Time128, TimeZone, to_local
    z = TimeZone("America/New_York")
    yield "ora", "now() @ z"            , benchmark(lambda: now() @ z)
    yield "ora", "to_local(now(), z)"   , benchmark(lambda: to_local(now(), z))
    yield "ora", "now(Time) @ z"        , benchmark(lambda: now(Time) @ z)
    yield "ora", "now(SmallTime) @ z"   , benchmark(lambda: now(SmallTime) @ z)
    yield "ora", "now(NsTime) @ z"      , benchmark(lambda: now(NsTime) @ z)
    yield "ora", "now(Time128) @ z"     , benchmark(lambda: now(Time128) @ z)


def benchmark_time_literal():
    from datetime import datetime
    yield "datetime", "datetime(…)"     , benchmark(lambda: datetime(2018, 1, 6, 16, 51, 45, 123456))

    import pytz
    z = pytz.timezone("America/New_York")
    yield "pytz", "z.localize(datetime(…))", benchmark(lambda: z.localize(datetime(2018, 1, 6, 16, 51, 45, 123456)))

    import dateutil.tz
    z = dateutil.tz.gettz("America/New_York")
    yield "dateutil", "datetime(…, z)"  , benchmark(lambda: datetime(2018, 1, 6, 16, 51, 45, 123456, z))

    from ora import Date, Daytime, Time, TimeZone, from_local, Jan
    z = TimeZone("America/New_York")
    yield "ora", "(Date(…), Daytime(…)) @ z", benchmark(lambda: (Date(2018, 1, 6), Daytime(16, 51, 45.123456)) @ z)
    yield "ora", "(Y/M/D, Daytime(…)) @ z", benchmark(lambda: (2018/Jan/6, Daytime(16, 51, 45.123456)) @ z)
    yield "ora", "from_local((Date(…), Time(…)), z)", benchmark(lambda: from_local((Date(2018, 1, 6), Daytime(16, 51, 45.123456)), z))
    yield "ora", "from_local((Date(…), Time(…)), '…')", benchmark(lambda: from_local((Date(2018, 1, 6), Daytime(16, 51, 45.123456)), "America/New_York"))
    yield "ora", "Time(…, z)"           , benchmark(lambda: Time(2018, 1, 6, 16, 51, 45.123456, z))
    yield "ora", "Time(…, '…')"         , benchmark(lambda: Time(2018, 1, 6, 16, 51, 45.123456, "America/New_York"))


def benchmark_convert_tz():
    import datetime
    import pytz
    z0 = pytz.timezone("America/New_York")
    z1 = pytz.timezone("Asia/Tokyo")
    t = datetime.datetime(2018, 1, 5, 21, 17, 56, 123456)
    yield "pytz", "z0.localize(t).astimezone(z1)", benchmark(lambda: z0.localize(t).astimezone(z1))

    import dateutil.tz
    z0 = dateutil.tz.gettz("America/New_York")
    z1 = dateutil.tz.gettz("Asia/Tokyo")
    t = datetime.datetime(2018, 1, 5, 21, 17, 56, 123456)
    yield "dateutil", "replace(tzinfo=z0).astimezone(z1)", benchmark(lambda: t.replace(tzinfo=z0).astimezone(z1))
    t = datetime.datetime(2018, 1, 5, 21, 17, 56, 123456, z0)
    yield "dateutil", "t.astimezone(z1)", benchmark(lambda: t.astimezone(z1))

    from ora import Date, Daytime, Time, TimeZone, to_local, from_local
    z0 = TimeZone("America/New_York")
    z1 = TimeZone("Asia/Tokyo")
    t = Date(2018, 1, 5), Daytime(21, 17, 56.123456)
    yield "ora", "t @ z0 @ z1"          , benchmark(lambda: t @ z0 @ z1)
    yield "ora", "to_local(from_local(t, z0), z1)", benchmark(lambda: to_local(from_local(t, z0), z1))


def benchmark_today_local():
    import datetime
    today = datetime.date.today
    yield "datetime", "date.today()"    , benchmark(lambda: today())

    from ora import today, get_display_time_zone, Date, Date16
    z = get_display_time_zone()
    yield "ora", "today(z)"            , benchmark(lambda: today(z))
    yield "ora", "today('display')"    , benchmark(lambda: today("display"))
    yield "ora", "today(z, Date)"      , benchmark(lambda: today(z, Date))
    yield "ora", "today('display', Date)", benchmark(lambda: today("display", Date))
    yield "ora", "today(z, Date16)"    , benchmark(lambda: today(z, Date16))
    yield "ora", "today('display', Date16)", benchmark(lambda: today("display", Date16))


def benchmark_time_format():
    import datetime
    t = datetime.datetime.utcnow()
    yield "datetime", "str(t)"          , benchmark(lambda: str(t))
    yield "datetime", "format(t, pyfmt)", benchmark(lambda: format(t, "%Y-%m-%d %H:%M:%S.%f"))
    yield "datetime", "format(t, rfc)"  , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%S.%fZ"))
    yield "datetime", "t.isoformat()"   , benchmark(lambda: t.isoformat())

    import ora
    from ora import format_time
    t = ora.now()
    yield "ora", "str(t)"               , benchmark(lambda: str(t))
    yield "ora", "format(t, pyfmt)"     , benchmark(lambda: format(t, "%Y-%m-%d %H:%M:%.6S"))
    yield "ora", "format(t, rfc)"       , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%.6SZ"))
    yield "ora", "format(t, dstrfc)"    , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%.6SZ@"))
    yield "ora", "format(t, tzrfc)"     , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%.6SZ@America/New_York"))
    yield "ora", "format(t, utcrfc)"    , benchmark(lambda: format(t, "%Y-%m-%dT%H:%M:%.6SZ@UTC"))
    yield "ora", "format_time(rfc, t)"  , benchmark(lambda: format_time("%Y-%m-%d", t))
    yield "ora", "format_time(rfc, t, '…')", benchmark(lambda: format_time("%Y-%m-%d", t, "America/New_York"))
    z = ora.TimeZone("America/New_York")
    yield "ora", "format_time(rfc, t, z)", benchmark(lambda: format_time("%Y-%m-%d", t, z))


def benchmark_date_format():
    import datetime
    d = datetime.date(2018, 1, 14)
    yield "datetime", "str(d)"          , benchmark(lambda: str(d))
    yield "datetime", "format(d, fmt)"  , benchmark(lambda: format(d, "%Y-%m-%d"))

    import ora
    d = ora.Date(2018, 1, 14)
    yield "ora", "str(d)"               , benchmark(lambda: str(d))
    yield "ora", "format(d, fmt)"       , benchmark(lambda: format(d, "%Y-%m-%d"))
    yield "ora", "format(d, tzfmt)"     , benchmark(lambda: format(d, "%Y-%m-%d@America/New_York"))


def benchmark_time_parse_iso_z():
    µs  = '2018-02-23T05:02:02.327973Z'
    s   = '2018-02-23T05:02:02Z'

    import datetime
    strptime = datetime.datetime.strptime
    yield "datetime", "strptime(s, ISOFMT_S)", benchmark(lambda: strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
    yield "datetime", "strptime(µs, ISOFMT_µS)", benchmark(lambda: strptime(µs, "%Y-%m-%dT%H:%M:%S.%fZ"))

    with suppress(ImportError):
        from udatetime import from_string
        yield "udatetime", "from_string(s)", benchmark(lambda: from_string(s))
        yield "udatetime", "from_string(µs)", benchmark(lambda: from_string(µs))

    with suppress(ImportError):
        from iso8601 import parse_date
        yield "iso8601", "parse_date(s)", benchmark(lambda: parse_date(s))
        yield "iso8601", "parse_date(µs)", benchmark(lambda: parse_date(µs))

    from numpy import datetime64
    yield "np", "datetime64(s)"         , benchmark(lambda: datetime64(s))
    yield "np", "datetime64(µs)"        , benchmark(lambda: datetime64(µs))
    yield "np", "datetime64(s, 's')"    , benchmark(lambda: datetime64(s, 's'))
    yield "np", "datetime64(µs, 'ns')"  , benchmark(lambda: datetime64(µs, 'ns'))

    from pandas import Timestamp
    yield "pd", "Timestamp(s)"          , benchmark(lambda: Timestamp(s))
    yield "pd", "Timestamp(µs)"         , benchmark(lambda: Timestamp(µs))

    from ora import Time, parse_time, parse_time_iso
    yield "ora", "parse_time(s, ISOFMT)", benchmark(lambda: parse_time("%Y-%m-%dT%H:%M:%S%e", s))
    yield "ora", "parse_time(µs, ISOFMT)", benchmark(lambda: parse_time("%Y-%m-%dT%H:%M:%S%e", µs))
    yield "ora", "parse_time(s, '%T')"  , benchmark(lambda: parse_time("%T", s))
    yield "ora", "parse_time(µs, '%T')" , benchmark(lambda: parse_time("%T", µs))
    yield "ora", "parse_time_iso(s)"    , benchmark(lambda: parse_time_iso(s))
    yield "ora", "parse_time_iso(µs)"   , benchmark(lambda: parse_time_iso(µs))
    yield "ora", "Time(s)"              , benchmark(lambda: Time(s))
    yield "ora", "Time(µs)"             , benchmark(lambda: Time(µs))


def benchmark_time_comparison():
    import datetime
    t0 = datetime.datetime.utcnow()
    t1 = datetime.datetime.utcnow()
    yield "datetime", "t0 == t1"        , benchmark(lambda: t0 == t1)
    yield "datetime", "t0 != t1"        , benchmark(lambda: t0 != t1)
    yield "datetime", "t0 <  t1"        , benchmark(lambda: t0 <  t1)
    yield "datetime", "t0 <= t1"        , benchmark(lambda: t0 <= t1)

    import ora
    t0 = ora.now()
    t1 = ora.now()
    yield "ora", "t0 == t1"             , benchmark(lambda: t0 == t1)
    yield "ora", "t0 != t1"             , benchmark(lambda: t0 != t1)
    yield "ora", "t0 <  t1"             , benchmark(lambda: t0 <  t1)
    yield "ora", "t0 <= t1"             , benchmark(lambda: t0 <= t1)


#-------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument(
    "category", metavar="CATEGORY", nargs="?", default=None,
    help="run CATEGORY benchmarks [def: all]")
args = parser.parse_args()

if args.category is None:
    fns = [ 
        (n[10 :], f)
        for n, f in globals().items() 
        if n.startswith("benchmark_") 
    ]
else:
    fns = [(args.category, globals()["benchmark_" + args.category])]

for cat, fn in fns:
    print(cat)
    print("-" * 63)
    for lib, name, elapsed in fn():
        print("{:10s} {:40s} {:8.3f} µs".format(lib, name, elapsed / 1e-6))
    print()


