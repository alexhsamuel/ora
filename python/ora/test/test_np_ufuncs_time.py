import numpy as np
import ora
from   ora import *
import pytest

pytest.importorskip("ora.np")

#-------------------------------------------------------------------------------

# FIXME: Combine.
TIME_TYPES = (SmallTime, Unix32Time, Unix64Time, Time, NsTime, HiTime, Time128)

# FIXME: Put the offset dtype in an attribute.
@pytest.mark.parametrize("Time,offset_dtype", [
    (SmallTime  , np.dtype("uint32")),
    (Unix32Time , np.dtype("int32")),
    (Unix64Time , np.dtype("int64")),
    (Time       , np.dtype("uint64")),
    (NsTime     , np.dtype("int64")),
    (HiTime     , np.dtype("uint64")),
])
def test_to_offset(Time, offset_dtype):
    arr = np.array([
        Time.MIN,
        Time.from_offset(0),
        Time.MAX,
        Time.INVALID,
        Time.MISSING,
    ])
    off = ora.np.to_offset(arr)
    assert off.dtype == offset_dtype
    assert len(off) == 5
    assert off[0] == Time.MIN.offset
    assert off[1] == 0
    assert off[2] == Time.MAX.offset


