from   time import perf_counter
from   datetime import datetime
import logging
import numpy as np
import ora
import pytz


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



def benchmark(fn, *, samples=20, n=5000, quantile=0.05):
    # Loop pedestal calculation.
    elapsed = []
    for _ in range(samples):
        start = perf_counter()
        for _ in range(n):
            pass
        elapsed.append(perf_counter() - start)
    null = np.percentile(elapsed, 100 * quantile, interpolation="nearest")
    
    elapsed = []
    for _ in range(samples):
        start = perf_counter()
        for _ in range(n):
            fn()
        elapsed.append(perf_counter() - start)
    
    elapsed = np.percentile(elapsed, 100 * quantile, interpolation="nearest")
    elapsed -= null
    return elapsed / n


def null_fn():
    pass


#-------------------------------------------------------------------------------

def benchmark_now():
    from ora import Time, SmallTime, NsecTime
    # from ora import NsTime
    yield "null"                , benchmark(lambda: None)
    yield "datetime.utcnow()"   , benchmark(lambda: datetime.utcnow())
    yield "datetime.now()"      , benchmark(lambda: datetime.now())
    yield "ora.now()"           , benchmark(lambda: ora.now())
    yield "ora.now(Time)"       , benchmark(lambda: ora.now(Time))
    yield "ora.now(SmallTime)"  , benchmark(lambda: ora.now(SmallTime))
    yield "ora.now(NsecTime)"   , benchmark(lambda: ora.now(NsecTime))
    # yield "ora.now(NsTime)"     , benchmark(lambda: ora.now(NsTime))



def summarize(benchmarks):
    for name, elapsed in benchmarks:
        print("{:32s} {:7.3f} µs".format(name, elapsed / 1e-6))


summarize(benchmark_now())


if False:
    print("convert time to string")

    time(lambda: none(), "null")

    x = datetime.utcnow()
    f = x.__str__
    u = time(lambda: f(), "datetime.datetime.__str__", unit=True)

    f = udatetime.to_string
    time(lambda: f(x), "udatetime.to_string", unit=u)

    x = delorean.Delorean.utcnow()
    f = x.format_datetime
    time(lambda: f("YYYY-mm-ddTHH:MM:SSZ"), "Delorean.format_datetime", unit=u)

    x = arrow.utcnow()
    f = x.__str__
    time(lambda: f(), "Arrow.__str__", unit=u)

    x = pendulum.utcnow()
    f = x.__str__
    time(lambda: f(), "Pendulum.__str__", unit=u)

    x = np.datetime64("now", "ns", unit=u)
    f = x.__str__
    time(lambda: f(), "np.datetime64.__str__", unit=u)

    x = pd.Timestamp.utcnow()
    f = x.__str__
    time(lambda: f(), "pd.Timestamp.__str__", unit=u)

    x = cron.now()
    f = x.__str__
    time(lambda: f(), "cron.Time.__str__", unit=u)

    print()


if False:
    print("convert from one time zone to another")

    time(lambda: none(), "null")
    
    z0 = pytz.timezone("America/New_York")
    x = z0.localize(datetime.now())
    z1 = pytz.timezone("Asia/Tokyo")
    f = x.astimezone
    u = time(lambda: f(z1), "datetime.astimezone pytz", unit=True)

    z0 = dateutil.tz.gettz("America/New_York")
    x = datetime.now().replace(tzinfo=z0)
    z1 = dateutil.tz.gettz("Asia/Tokyo")
    f = x.astimezone
    time(lambda: f(z1), "datetime.astimezone dateutil.tz", unit=u)

    x = delorean.Delorean.now("America/New_York")
    f = x.shift
    time(lambda: f("Asia/Tokyo"), "Delorean.shift", unit=u)

    x = arrow.now()
    z1 = pytz.timezone("Asia/Tokyo")
    f = x.to
    time(lambda: f(z1), "Arrow.to pytz", unit=u)

    x = arrow.now()
    z1 = dateutil.tz.gettz("Asia/Tokyo")
    f = x.to
    time(lambda: f(z1), "Arrow.to dateutil.tz", unit=u)

    x = pendulum.now()
    z1 = pytz.timezone("Asia/Tokyo")
    f = x.in_timezone
    time(lambda: f(z1), "Pendulum.in_timezone pytz", unit=u)
    
    # x = pendulum.now()
    # z1 = dateutil.tz.gettz("Asia/Tokyo")
    # f = x.in_timezone
    # time(lambda: f(z1), "Pendulum.in_timezone dateutil.tz", unit=u)
    
    x = pd.Timestamp.now("America/New_York")
    z1 = pytz.timezone("Asia/Tokyo")
    f = x.astimezone
    time(lambda: f(z1), "pd.Timestamp.astimezone pytz", unit=u)

    x = pd.Timestamp.now("America/New_York")
    z1 = dateutil.tz.gettz("Asia/Tokyo")
    f = x.astimezone
    time(lambda: f(z1), "pd.Timestamp.astimezone dateutil", unit=u)

    x = cron.now()
    z = cron.TimeZone("Asia/Tokyo")
    f = cron.to_local
    time(lambda: f(x, z), "cron.to_local", unit=u)

    x = cron.now()
    z = cron.TimeZone("Asia/Tokyo")
    time(lambda: x @ z, "cron @", unit=u)

    print()


if False:
    print("get minute of local time")

    time(lambda: None, "null")
    
    z = pytz.timezone("America/New_York")
    x = z.localize(datetime.now())
    u = time(lambda: x.minute, "datetime.minute", unit=True)

    x = delorean.Delorean.now("America/New_York")
    time(lambda: x.datetime.minute, "Delorean.datetime.minute", unit=u)

    x = arrow.now("America/New_York")
    time(lambda: x.minute, "Arrow.minute", unit=u)

    x = pendulum.now("America/New_York")
    time(lambda: x.minute, "Pendulum.minute", unit=u)

    x = pd.Timestamp.now("America/New_York")
    time(lambda: x.minute, "Timestamp.minute", unit=u)

    x = cron.now()
    z = cron.TimeZone("America/New_York")
    time(lambda: (x @ z).daytime.minute, "cron @ .daytime.minute", unit=u)

    print()


if False:
    print("parse UTC time")
    s = '2017-11-23T19:11:00.080593Z'
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"

    f = datetime.strptime
    u = time(lambda: f(s, fmt), "datetime.strptime", unit=True)

    f = udatetime.from_string
    time(lambda: f(s), "udatetime.from_string", unit=u)

    f = delorean.parse
    time(lambda: f(s), "delorean.parse", unit=u)

    f = arrow.get
    time(lambda: f(s), "arrow.get", unit=u)

    f = pendulum.from_format
    time(lambda: f(s, fmt), "pendulum.from_format", unit=u)

    f = pendulum.strptime
    time(lambda: f(s, fmt), "pendulum.strptime", unit=u)

    f = pendulum.parse
    time(lambda: f(s), "pendulum.parse", unit=u)

    print()



