#include <iostream>

#include "cron/ez.hh"
#include "cron/format.hh"
#include "cron/localization.hh"
#include "cron/time.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::ez;
using namespace cron::daytime;
using namespace cron::time;

//------------------------------------------------------------------------------

TEST(Time, mins) {
  EXPECT_EQ(Time::MIN       , from_local            (1, 0, 0, 0, 0, 0, UTC));
  EXPECT_EQ(Time::MIN       , from_local<Time>      (1, 0, 0, 0, 0, 0, UTC));
  EXPECT_EQ(Unix32Time::MIN , from_local<Unix32Time>(1901, 11, 12, 20, 45, 52, UTC));
  EXPECT_EQ(Unix64Time::MIN , from_local<Unix64Time>(1, 0, 0, 0, 0, 0, UTC));
  EXPECT_EQ(SmallTime::MIN  , from_local<SmallTime> (1970, 0, 0, 0, 0, 0, UTC));
  EXPECT_EQ(Time128::MIN    , from_local<Time128>   (1, 0, 0, 0, 0, 0, UTC));
}

TEST(Time, maxs) {
  EXPECT_EQ(Time::MAX       , from_local            (8711, 6, 15, 6, 9, 3.99999995, UTC));
  EXPECT_EQ(Time::MAX       , from_local<Time>      (8711, 6, 15, 6, 9, 3.99999995, UTC));
  EXPECT_EQ(Unix32Time::MAX , from_local<Unix32Time>(2038, 0, 18, 3, 14, 5, UTC));
  EXPECT_EQ(Unix64Time::MAX , from_local<Unix64Time>(9999, 11, 30, 23, 59, 59, UTC));
  EXPECT_EQ(SmallTime::MAX  , from_local<SmallTime> (2106, 1, 6, 6, 28, 13, UTC));

  auto max128 = from_local<Time128>(9999, 11, 30, 23, 59, 59.99999999999, UTC);
  // FIXME: Difference should not be zero!
  EXPECT_TRUE(std::abs(Time128::MAX - max128) < 1E-12);
}

TEST(Time, from_utc) {
  auto const d = 2016/MAY/29;
  Daytime const y(23, 30, 15.5);
  EXPECT_EQ(from_local(d, y, UTC), from_utc(d, y));

  EXPECT_EQ(from_local<Unix32Time>(d, y, UTC), from_utc<Unix32Time>(d, y));
  EXPECT_EQ(from_local<Unix64Time>(d, y, UTC), from_utc<Unix64Time>(d, y));
  EXPECT_EQ(from_local<SmallTime>(d, y, UTC), from_utc<SmallTime>(d, y));
  EXPECT_EQ(from_local<Time128>(d, y, UTC), from_utc<Time128>(d, y));
}

