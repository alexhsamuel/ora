# Setup

To build and test Ora, you'll require:
- A C++14 compiler, such as GCC >= 6
- Python 3.6
- NumPy
- pytz (for testing)
- pytest (for testing)

After you clone the repo, you'll have to unpack the zoneinfo database:
```
$ make share/zoneinfo
```


# Developing



# Packaging

Ora packages a copy of the zoneinfo database with its Python package.  The
zoneinfo files are read from C++ code.  For this reason, Ora cannot be packaged
as an egg.

