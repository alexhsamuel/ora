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
from   .ext import set_up_numpy as _set_up_numpy

# Add all the numpy stuff to the extension module.
_set_up_numpy()

from   .ext import day, month, year, ymdi
