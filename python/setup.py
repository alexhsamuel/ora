from   glob import glob
import os
from   setuptools import setup, Extension
import sys

if sys.platform == "darwin":
  # No C++14 when building for earlier OSX versions.
  os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"

setup(
  name="test",
  ext_modules=[
    Extension(
      "cron._ext",
      extra_compile_args=["-std=c++14", ],
      include_dirs      =["../include", ],
      sources           =glob("cron/*.cc"),
      library_dirs      =["../lib",],
      libraries         =["cron",],
      depends           =glob("cron/*.hh") + glob("../include/*.hh"),
    ),

  ]
)

