# Requirements

To build and test Ora, you'll require:
- A C++14 compiler, such as GCC >= 6
- Python 3.13
- NumPy (optional)
- pytz (for testing)
- pytest (for testing)


# Developing

Add the `python/` subdirectory to your `PYTHONPATH`.

To build C++ and Python libraries: `make`.

To run C++ and Python tests, `make test`.

You may also use `setup.py` to build, and `pytest` to run Python tests.


# Testing

Run exhaustive tests against `zoneinfo` with `pytest -m fulltz`.



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
uv pip install bump-my-version
bump-my-version bump patch  # or minor, major
git push --tags
git push
python setup.py sdist
twine upload dist/ora-...tar.gz
```

### OS/X binary wheel and conda package

```
make clean; rm -rf build
python setup.py bdist_wheel
conda build conda-recipe --python 3.6
anaconda upload ...
```

### Linux conda package

Builds a conda (binary) package on CentOS7.

```
docker run -ti --rm conda/miniconda3-centos7 bash
yum group install -y "Development Tools"
yum install -y centos-release-scl
yum install -y devtoolset-7
export PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH
conda install -y anaconda-client conda-build
git clone https://github.com/alexhsamuel/ora
conda build ora/conda-recipe
anaconda upload ...
```

### Linux binary wheel

To build and upload a binary wheel (requires Podman):
```
$ ./build-manylinux-wheel
$ twine upload ora-*-manylinux*.whl
```

See the script to change the build image or target Python version.


# Talks

Talks and other media are on the `gh-pages` branch and available at 
http://alexhsamuel.github.io.


