from   glob import glob
import os
from   setuptools import setup, Extension
import sys

if sys.platform == "darwin":
    # No C++14 when building for earlier OSX versions.
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"

include_dirs = ["../include", ]

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
            "cron._ext",
            extra_compile_args=[
                "-std=c++14", 
                "-fdiagnostics-color=always", 
                "-O0", 
            ],
            include_dirs      =include_dirs,
            sources           =glob("cron/*.cc"),
            library_dirs      =["../lib",],
            libraries         =["cron",],
            depends           =glob("cron/*.hh") + glob("../include/*.hh"),
        ),

  ]
)

