import numpy as np
import pytest

import ora
from   ora import *

DAYTIMES = (Daytime, Daytime32, UsecDaytime)

#-------------------------------------------------------------------------------

@pytest.mark.parametrize("Daytime", DAYTIMES)
def test_daytime_array(Daytime):
    arr = np.array(
        [Daytime(0, 0), Daytime(14,31, 25), Daytime.INVALID, Daytime.MISSING])
    assert arr.dtype == Daytime.dtype



