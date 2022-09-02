

# Special calendars:

- `all`
- `none`
- weekday: `Mon` - `Sun`
- fiscal quarters:
  - `Q` = `Q/Jan`, quarters with Q1 starting in Jan
- `M[2]`: second day of each month
- `M[2,Mon]`: second Monday of each month
- `Y[-1]`: last day of each year
- `Jan[-1]`: last day of each Jan
- `Q/Oct[-2]`: second day before the end of each Oct-starting quarter
- `Q2[-1]`: last day of each (Jan-starting) Q2
- halves

# Operations on calendars

- `~`: invert calendar
- `|`, `&`, `-`, `^`: set ops

- `cal - 1` `cal + 1`: physical date shift

- `cal0 << cal1`: each day in cal0, if it's in cal1, or else the previous date in cal1 (remove dups)
  e.g. `M[1] >> US/federal/businessdays`: first Mon of month, or next US federal workday

- `cal0 <<< cal1`: noninclusive before

- `shift(cal0, cal1, n)`: each date in `cal0`, shifted `n` days forward in `cal1`


# Opertions on dates

### Some other options

```
cal(date + 0)
cal(date - 2)

date + 0 in cal
date next in cal = date + 0 in cal
date - 2 in cal
```

