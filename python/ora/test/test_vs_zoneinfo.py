import datetime
import ora
import pytest
import sys

#-------------------------------------------------------------------------------

@pytest.mark.skipif(sys.version_info < (3, 9), reason="no zoneinfo")
@pytest.mark.parametrize(
    "name",
    sorted(ora.list_zoneinfo_dir())
)
def test_zones(name):
    import zoneinfo
    z0 = zoneinfo.ZoneInfo(name)
    z1 = ora.TimeZone(name)

    for year in range(1950, 2050):
        for month in range(1, 13):
            for days_off in (-10, 0, 10):
                t0 = datetime.datetime(year, month, 1, 0, 0, 0) + datetime.timedelta(days_off)
                o0 = z0.utcoffset(t0).total_seconds()

                t1 = ora.Date(year, month, 1) + days_off, ora.MIDNIGHT
                try:
                    o1 = z1.at_local(t1).offset
                except ValueError:
                    # This local time doesn't exist.
                    continue

                assert o1 == o0, f"mismatch: {name} {t1}: {o0} {o1}"


