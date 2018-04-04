import numpy as np
import pytest

import ora
from   ora import *

#-------------------------------------------------------------------------------

def get_array(Daytime):
    return np.array(
        [Daytime(0, 0), Daytime(14,31, 25), Daytime.INVALID, Daytime.MISSING])


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_daytime_array(Daytime):
    arr = get_array(Daytime)
    assert arr.dtype == Daytime.dtype


@pytest.mark.parametrize("Daytime", DAYTIME_TYPES)
def test_is_valid(Daytime):
    arr = get_array(Daytime)
    assert (ora.np.is_valid(arr).astype(int) == [1, 1, 0, 0]).all()


