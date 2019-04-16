import numpy as np
from   ora import Time, Unix32Time, UTC
import ora.np

#-------------------------------------------------------------------------------

def test_time_from_offset():
    offset = (1 << np.arange(30)).reshape(3, 5, 2)
    time = ora.np.time_from_offset(offset, dtype=Unix32Time)
    assert time.shape == (3, 5, 2)
    assert time.dtype == Unix32Time.dtype
    assert time[0][0][0] == Unix32Time.from_offset(1)
    assert time[2][4][1] == Unix32Time.from_offset(1 << 29)


def test_time_from_offset_cast():
    offset = np.array([0, 1, 60], dtype="uint32") * Time.DENOMINATOR
    time = ora.np.time_from_offset(offset)
    assert time.shape == (3, )
    assert time.dtype == Time.dtype
    assert all(time == [
        Time(1, 1, 1, 0, 0, 0, UTC),
        Time(1, 1, 1, 0, 0, 1, UTC),
        Time(1, 1, 1, 0, 1, 0, UTC),
    ])


