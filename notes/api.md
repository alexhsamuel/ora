# C++

## Date

```py
ymdi(date)
ymdi(date_arr)
```

```py
date = Date.from_ymdi(i)
date_arr = from_ymdi(i_arr, dtype=Date.dtype)
```

This is annoying for two reasons:

1. To implement it efficiently, we'll need to recreate some kind of loop 
function lookup for `from_ymdi()` based on the dtype.

1. We'll need to install a replacement `from_ymdi()` function that handles
scalars and arrays, at numpy load time.

