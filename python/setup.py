from   glob import glob
import os
from   setuptools import setup, Extension
import sys

if sys.platform == "darwin":
    # No C++14 when building for earlier OSX versions.
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"

include_dirs = ["../c++/include", ]

#-------------------------------------------------------------------------------

# FIXME: We should just require numpy to build, no?

try:
    import numpy
except ImportError:
    have_numpy = False
    print("no numpy found; building without")
else:
    have_numpy = True
    from numpy.distutils.misc_util import get_numpy_include_dirs
    include_dirs.extend(get_numpy_include_dirs())

#-------------------------------------------------------------------------------

setup(
    name="test",
    ext_modules=[
        Extension(
            "ora.ext",
            extra_compile_args=[
                "-std=c++14", 
                "-fdiagnostics-color=always", 
                "-O0", 
            ],
            include_dirs      =include_dirs,
            sources           =glob("ora/*.cc"),
            library_dirs      =["../c++/lib",],
            libraries         =["ora",],
            depends           =glob("ora/*.hh") + glob("../c++/include/*.hh"),
        ),

  ]
)

