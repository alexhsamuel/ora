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

    local_times = (
        (ora.Date(year, month, 1) + day_off, ora.MIDNIGHT + sec_off)
        for year in range(1950, 2050)
        for month in range(1, 13)
        for day_off in (-10, 0, 10)
        for sec_off in (-14400, 0, 7200, 9000, 43200)
    )

    for local_time in local_times:
        t0 = (local_time @ ora.UTC).std.replace(tzinfo=None)
        o0 = z0.utcoffset(t0).total_seconds()

        try:
            o1 = z1.at_local(local_time).offset
        except ValueError:
            # This local time doesn't exist.
            continue

        assert o1 == o0, f"mismatch: {name} {t1}: {o0} {o1}"


