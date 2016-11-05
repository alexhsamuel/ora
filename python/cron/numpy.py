"""
Numpy support for Cron.

Numpy support is provided in a submodule so that users who do not need it aren't
forced to import numpy.

After this module is imported, Cron types that can be used as numpy array types
will have a `dtype` attribute.  For example:

  >>> import numpy as np
  >>> from cron import Date
  >>> import cron.numpy
  >>> array = np.zeros(3, dtype=Date.dtype)
  >>> array
  array([Date(1, Jan, 1), Date(1, Jan, 1), Date(1, Jan, 1)], dtype=Date)

"""

#-------------------------------------------------------------------------------

import numpy
from   . import ext

# Add all the numpy stuff to the extension module.
ext.set_up_numpy()

# We can't "from .ext.numpy import *" since .ext isn't a package.
globals().update({ 
    n: o 
    for n, o in ext.numpy.__dict__.items() 
    if not n.startswith("_") 
})

del ext
