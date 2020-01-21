"""
NumPy support for Ora.

Ora types can be used as numpy array types, and will have a `dtype` attribute.
For example:

  >>> import numpy as np
  >>> from ora import Date
  >>> import ora.np
  >>> array = np.zeros(3, dtype=Date.dtype)
  >>> array
  array([Date(1, Jan, 1), Date(1, Jan, 1), Date(1, Jan, 1)], dtype=Date)

"""

#-------------------------------------------------------------------------------

from   .. import ext

try:
    ext.np
except AttributeError:
    # Not built with numpy support.
    raise ImportError("Ora not build with NumPy support")
else:
    # We can't "from .ext.np import *" since .ext isn't a package.
    globals().update({ 
        n: o 
        for n, o in ext.np.__dict__.items() 
        if not n.startswith("_") 
    })

    del ext

