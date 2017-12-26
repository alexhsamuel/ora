This document describes the Python API to Cron.  Please see [ontology.md](ontology.md) for a description of the main concepts and terms in Cron, and their semantics.

The code below assumes the following import, but of course feel free to use qualified names in your programs.

```py
>>> from ora import *
```

# Types

The C++ types for times, dates, and daytimes are templated; Python, however, does not support templates, as they are a compile-time construct.  Instead, The Python extension module contains a variety of independent extension types that wrap various instances of the C++ templates.  Each of these is an independent type, but with the same API as the other variants.

- [dates](python-date.md)
- [daytimes](python-daytime.md)
- [time zones](python-time-zone.md)
- [times](python-time.md)
- [localizing times](python-local.md)

