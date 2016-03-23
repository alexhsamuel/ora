"""
Numpy support for Cron.

Numpy support is provided in a submodule so that users who do not need it aren't
forced to import numpy.

After this module is imported, Cron types that can be used as numpy array types
will have a `dtype` attribute.  For example:

  >>> import numpy as np
  >>> from cron import Date
  >>> import cron.numpy
  >>> array = np.zeroes(8, dtype=Date.dtype)

"""

#-------------------------------------------------------------------------------

import numpy
from   . import _ext

# Adds
_ext.set_up_numpy()

from   ._ext import year, month, day
