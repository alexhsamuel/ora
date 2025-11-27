"""
Ora is a freestanding time and date implementation for C++ and Python.

Ora is `hosted on GitHub <http://github.com/alexhsamuel/ora>`_.  See
the `installation instructions
<https://github.com/alexhsamuel/ora/blob/master/README.md#installation>`_.

Docs at `readthedocs <http://ora.readthedocs.io/en/latest/>`_.
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

if sys.platform == "darwin":
    # No C++14 when building for earlier OSX versions.
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"


#-------------------------------------------------------------------------------

def np_get_include():
    try:
        import numpy
        return [numpy.get_include()]
    except ImportError:
        return []


# Convince setuptools to call our C++ build.

class BuildExt(setuptools.command.build_ext.build_ext):

    def run(self):
        try:
            import numpy
        except ImportError:
            py_np = False
        else:
            py_np = True
        env = os.environ | {"PY_NP": "yes" if py_np else "no"}
        subprocess.check_call(["make", "cxx", "docstrings"], env=env)
        setuptools.command.build_ext.build_ext.run(self)



#-------------------------------------------------------------------------------

setup(
    name            ="ora",
    version         ="0.10.0",
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],

    python_requires='>=3.6',
    extras_require={
        "numpy": ["numpy<2"],
    },
    package_dir     ={"": "python"},
    packages        =["ora", "ora.np"],
    package_data    ={
        ""      : ["test/*"],
        "ora"   : ["calendars/*", "zoneinfo/*", "zoneinfo/*/*"],
    },

    ext_modules=[
        Extension(
            "ora.ext",
            extra_compile_args=[
                "-std=c++17",
                "-fdiagnostics-color=always",
                "-Wno-dangling-else",
            ],
            include_dirs      =[
                "cxx/include",
                "python/ora/ext",
                *np_get_include(),
            ],
            sources           =glob("python/ora/ext/*.cc"),
            library_dirs      =["cxx/src", ],
            libraries         =["ora", ],
            depends           =[
                *glob("cxx/include/*.hh"),
                *glob("python/ora/ext/*.hh"),
            ]
        ),
    ],

    cmdclass={
        "build_ext" : BuildExt,
    },
)

