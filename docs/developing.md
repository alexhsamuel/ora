# Setup

To build and test Ora, you'll require:
- A C++14 compiler, such as GCC >= 6
- Python 3.6
- NumPy
- pytz (for testing)
- pytest (for testing)

To run tests, build Google's gtest (one time only):

```sh
cd external/gtest
make gtest_main.a
```

To unpack the time zone data (one time only):

```
make zoneinfo
```


# Developing

WRITEME.


# Packaging

Ora packages a copy of the zoneinfo database with its Python package.  The
zoneinfo files are read from C++ code.  For this reason, Ora cannot be packaged
as an egg.


