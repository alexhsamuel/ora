#include "ora.hh"
#include "gtest/gtest.h"

using namespace ora::lib;
using namespace ora;
using namespace ora::ez;

//------------------------------------------------------------------------------

TEST(Time, basic) {
  auto const time = from_utc_parts(2016, 5, 30, 20, 15, 0);

  auto const nyc = to_local(time, "America/New_York");
  EXPECT_EQ(2016/MAY/30, nyc.date);
  EXPECT_EQ(from_hms(16, 15), nyc.daytime);

  auto const utc = to_utc(time);
  EXPECT_EQ(2016/MAY/30, utc.date);
  EXPECT_EQ(from_hms(20, 15), utc.daytime);

  auto const tokyo = to_local(time, "Asia/Tokyo");
  EXPECT_EQ(2016/MAY/31, tokyo.date);
  EXPECT_EQ(from_hms(5, 15), tokyo.daytime);
}

