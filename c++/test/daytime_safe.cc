#include "gtest/gtest.h"
#include "cron/daytime.hh"
#include "cron/format.hh"

using namespace cron;
using namespace cron::daytime;

//------------------------------------------------------------------------------

TEST(Daytime, from_offset) {
  EXPECT_EQ(Daytime::INVALID, safe::from_offset(-1));
  EXPECT_EQ(Daytime::MIN, safe::from_offset(0));
  EXPECT_EQ(from_offset(10000), safe::from_offset(10000));
  EXPECT_EQ(Daytime::INVALID, safe::from_offset(Daytime::MAX.get_offset() + 1));
}

TEST(Daytime, from_daytick) {
  EXPECT_EQ(Daytime::INVALID, safe::from_daytick(-1));
  EXPECT_EQ(Daytime::MIN, safe::from_daytick(0));
  EXPECT_EQ(from_daytick(10000000), safe::from_daytick(10000000));
  EXPECT_EQ(Daytime::INVALID, safe::from_daytick(Daytime::MAX.get_daytick() + 1));
}

TEST(Daytime, from_hms) {
  EXPECT_EQ(Daytime::MIDNIGHT, safe::from_hms(0, 0));
  EXPECT_EQ(Daytime::MIDNIGHT, safe::from_hms(0, 0, 0));

  for (Hour h = 0; h < 24; ++h)
    for (Minute m = 0; m < 60; m += 5)
      for (Second s = 0; s < 60; s += 11)
        EXPECT_EQ(from_hms(h, m, s), safe::from_hms(h, m, s));
}

TEST(Daytime, from_hms_invalid) {
  EXPECT_EQ(Daytime::INVALID, safe::from_hms(-1,  0,  0));
  EXPECT_EQ(Daytime::INVALID, safe::from_hms(24,  0,  0));
  EXPECT_EQ(Daytime::INVALID, safe::from_hms( 0, -1,  0));
  EXPECT_EQ(Daytime::INVALID, safe::from_hms( 0, 60,  0));
  EXPECT_EQ(Daytime::INVALID, safe::from_hms( 0,  0, -1));
  EXPECT_EQ(Daytime::INVALID, safe::from_hms( 0,  0, 60));
}

TEST(Daytime, from_hms_struct) {
  struct HmsDaytime hms;
  EXPECT_EQ(Daytime::INVALID, safe::from_hms(hms));
  hms.hour = 0;
  EXPECT_EQ(Daytime::INVALID, safe::from_hms(hms));
  hms.minute = 0;
  EXPECT_EQ(Daytime::INVALID, safe::from_hms(hms));
  hms.second = 0;
  EXPECT_EQ(Daytime::MIDNIGHT, safe::from_hms(hms));

  hms = HmsDaytime{23, 59, 59.99609375};
  EXPECT_EQ(from_hms(23, 59, 59.99609375), safe::from_hms(hms));
}

