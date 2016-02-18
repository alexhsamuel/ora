# Setup

To run tests, build Google's gtest (one time only):

```sh
cd test/gtest
make gtest_main.a
```

To unpack the time zone data (one time only):

```
make zoneinfo
```

# Comparison with `std::chrono`

- chrono splits out the duration from the epoch
- chrono uses a nice `ratio` class.
- chrono mixes `now` into the clock description.
- chrono provides user-defined literals
- chorno doesn't provide time of day or date types
- chrono doesn't provide NaT / invalid (good or bad?)
