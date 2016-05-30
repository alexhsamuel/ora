#include "cron/ez.hh"
#include "cron/format.hh"
#include "cron/localization.hh"
#include "cron/time.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::ez;
using namespace cron::time;
using cron::daytime::Daytime;

//------------------------------------------------------------------------------

TEST(Time128, convert_to) {
  Time const t0 = from_utc_parts(2016, 4, 26, 16, 13, 26.577521);
  Time128 const t = t0;
  EXPECT_TRUE(t.is_valid());

  auto parts = get_parts(t, UTC);
  EXPECT_EQ(2016, parts.date.year);
  EXPECT_EQ(   4, parts.date.month);
  EXPECT_EQ(  26, parts.date.day);
  EXPECT_EQ(  16, parts.daytime.hour);
  EXPECT_EQ(  13, parts.daytime.minute);
  EXPECT_TRUE(std::abs(parts.daytime.second - 26.577521) < 1e-6);
}

