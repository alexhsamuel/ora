## Python Localizing

The central feature of Cron is converting between a time, which is an abstract specification of a particular physical instant, and one of its local representations.  A local representation is the date and daytime in a particular time zone which specifies that time.

### Time from local representation

The `from_local()` function converts a (date, daytime) pair and time zone to a time.  This, for example, is when Apollo 11 landed on the moon.

```py
>>> landing = from_local((1969/Jul/20, Daytime(20, 18, 4)), UTC)
```

If you were to print `landing`, you'd see the same UTC date and daytime.  That's because Cron has to print _something_; for humans, date and time are the commonly accepted way of specifying a time.  Internally, however, Cron does not store date and daytime.  It converts to these when it needs to print out the time.
 

### Time to local representation

The `to_local()` function is the inverse of `from_local()`.  It takes a time and time zone, and returns the corresponding date and daytime in that time zone.

```py
>>> date, daytime = to_local(landing, "US/Central")
>>> print(date, daytime)
1969-07-20 15:18:04.000000000000000
```

