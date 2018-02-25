# Requirements

To build and test Ora, you'll require:
- A C++14 compiler, such as GCC >= 6
- Python 3.6
- NumPy
- pytz (for testing)
- pytest (for testing)


# Developing

Add the `python/` subdirectory to your `PYTHONPATH`.

To build C++ and Python libraries: `make`.

To run C++ and Python tests, `make test`.

You may also use `setup.py` to build, and `pytest` to run Python tests.


# Benchmarks

Run [`benchmarks.py`](/benchmarks/benchmark.py) for some performace tests,
including comparisons to other time and date libraries.


# Zoneinfo data

The `share/zoneinfo` directory contains time zone information.  It is compiled
from information in the "Olsen" [tz
database](http://web.cs.ucla.edu/~eggert/tz/tz-link.htm) and its contents are in
the public domain.  The data are updated several times a year to reflect
political changes in time zone policy.

On UNIX-like systems, a version is generally installed in
`/usr/share/zoneinfo`.  We include another copy here as the system copy might
not be up to date, and may also employ older formats for the data files (as is
the case with Darwin/OSX).

To update the zoneinfo data to a new tz database release:
```
external/update-zoneinfo.sh VERSION
```
and commit the updated files.

The C++ time zone code consults the `ZONEINFO` environment variable for the
location of the zoneinfo files.  If this is not set, the default
`/usr/share/zoneinfo` is used.  To set the zoneinfo directory location in code,
use `ora::set_zoneinfo_dir()`.

In Python, we install the zoneinfo data along with the code, and Ora by default
uses its own zoneinfo data.  Use `ora.set_zoneinfo_dir()` to change this.


# Packaging

Ora packages a copy of the zoneinfo database with its Python package.  The
zoneinfo files are read from C++ code.  For this reason, Ora cannot be packaged
as an egg.


# Releasing

### Source:

```
bumpversion patch  # or minor, major
git push --tags
git push
make clean
python setup.py sdist upload
```

### OS/X binary

```
python setup.py bdist_wheel upload
conda build conda-recipe --python 3.6
anaconda upload ...
```

### Linux binary

```
docker run -ti --rm continuumio/conda_builder_linux bash
git clone https://github.com/alexhsamuel/ora
conda build ora/conda-recipe --python 3.6
anaconda upload ...
```

