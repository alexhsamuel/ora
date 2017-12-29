"""
Ora is a coherent and high-perforamance C++ and Python 3 library for times,
dates, time zones, and related concepts.  The central concept is a
location-independent instant of time.  Local date and time of day
representations are derived from this, using the (included) zoneinfo database.
Multiple resolutions are provided for all time types.

.. code:: py

    >>> import ora
    >>> time = ora.now()
    >>> print(time)
    2017-12-26T03:47:36.41359399Z
    >>> tz = ora.TimeZone("US/Eastern")
    >>> (time @ tz).date
    Date(2017, Dec, 25)

"""

#-------------------------------------------------------------------------------

from   glob import glob
import os
from   setuptools import setup, Extension
import setuptools.command.build_ext
import setuptools.command.install
import subprocess
import sys

#-------------------------------------------------------------------------------

# FIXME: We should just require numpy to build, no?

try:
    import numpy
except ImportError:
    have_numpy = False
    numpy_include_dirs = []
    print("no numpy found; building without")
else:
    have_numpy = True
    from numpy.distutils.misc_util import get_numpy_include_dirs
    numpy_include_dirs = get_numpy_include_dirs()

#-------------------------------------------------------------------------------

if sys.platform == "darwin":
    # No C++14 when building for earlier OSX versions.
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"


#-------------------------------------------------------------------------------

# Convince setuptools to call our C++ build.

class BuildExt(setuptools.command.build_ext.build_ext):

    def run(self):
        subprocess.check_call(["make", "cxx", "docstrings"])
        setuptools.command.build_ext.build_ext.run(self)



class Install(setuptools.command.install.install):

    def run(self):
        # subprocess.check_call(["make", "install"])
        setuptools.command.install.install.run(self)



#-------------------------------------------------------------------------------

setup(
    name            ="ora",
    version         ="0.1.2",
    description     ="Alternative time and date library",
    long_description=__doc__,
    url             ="https://github.com/alexhsamuel/ora",
    author          ="Alex Samuel",
    author_email    ="alexhsamuel@gmail.com",
    license         ="BSD-3",
    keywords        =["time", "date", "time zone"],
    classifiers     =[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],    

    # FIXME: Relax this.
    requires        =[
        "numpy",
    ],

    package_dir={"": "python"},
    packages=["ora"],
    package_data={"": ["test/*"]},

    ext_modules=[
        Extension(
            "ora.ext",
            extra_compile_args=[
                "-std=c++14", 
                "-fdiagnostics-color=always", 
                "-O0", 
            ],
            include_dirs      =["cxx/include"] + numpy_include_dirs,
            sources           =glob("python/ora/*.cc"),
            library_dirs      =["cxx/src",],
            libraries         =["ora",],
            depends           =glob("python/ora/*.hh") + glob("cxx/include/*.hh"),
        ),
    ],

    cmdclass={
        "build_ext" : BuildExt,
        "install"   : Install,
    },
)

