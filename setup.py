"""
Ora is a freestanding time and date implementation for C++ and Python.

Ora is `hosted on GitHub <http://github.com/alexhsamuel/ora>`_.  See
the `installation instructions
<https://github.com/alexhsamuel/ora/blob/master/README.md#installation>`_.

Docs at `readthedocs <http://ora.readthedocs.io/en/latest/>`_.
"""

#-------------------------------------------------------------------------------

from   glob import glob
import numpy as np
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

# Convince setuptools to call our C++ build.

class BuildExt(setuptools.command.build_ext.build_ext):

    def run(self):
        subprocess.check_call(["make", "cxx", "docstrings"])
        setuptools.command.build_ext.build_ext.run(self)



#-------------------------------------------------------------------------------

def enumerate_data_files(dir):
    """
    Enumerates files suitable for setuptools's `data_files` option.

    Generates (dir_path, file_paths) pairs for each directory under `dir`,
    where `file_paths` is a list of paths to files in that directory.
    """
    for dir, _, files in os.walk(dir):
        yield dir, [ os.path.join(dir, f) for f in files ]


setup(
    name            ="ora",
    version         ="0.7.1",
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

    python_requires='>=3.6',
    setup_requires=[
        "numpy",
    ],
    install_requires=[
        "numpy",  # Required to install, but not to use.
    ],

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
                np.get_include(),
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

