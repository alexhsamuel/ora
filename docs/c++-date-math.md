# C++ date math

```c++
#include "cron.hh"
using namespace cron;
```

## Weekday conventions

There are a number of enumerations in wide use for representing weekdays.  For
example, 

- Cron uses Monday = 0 through Sunday = 6.
- The ISO 8601 standard specifies Monday = 1 through Sunday = 7.
- The C library's `struct tm` uses Sunday = 0 through Saturday = 6.

`weekday::Convention` provides provides conversions among these.  The class
is templated, but type aliases are provided for the cases above.  The `encode()`
method convers a `Weekday` to an integer with the specified convention, and
`decode()` reverses this.

For example,

```c++
int w = weekday:ENCODING_ISO::encode(TUESDAY);
Weekday w = weekday::ENCODING_CLIB::decode(4);
```

