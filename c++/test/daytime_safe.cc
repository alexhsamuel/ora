#include "gtest/gtest.h"
#include "cron/daytime.hh"
#include "cron/format.hh"

using namespace cron;
using namespace cron::daytime;

//------------------------------------------------------------------------------

inline bool
check(
  HmsDaytime const& hms,
  Hour const hour,
  Minute const minute,
  Second const second)
{
  return hms.hour == hour && hms.minute == minute && hms.second == second;
}


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

TEST(Daytime, from_ssm) {
  EXPECT_EQ(Daytime::INVALID                , safe::from_ssm(-0.01));
  EXPECT_EQ(Daytime::MIDNIGHT               , safe::from_ssm(0));
  EXPECT_EQ(from_hms(23, 59, 59)            , safe::from_ssm(86399));
  EXPECT_EQ(from_hms(23, 59, 59.99609375)   , safe::from_ssm(86399.99609375));
  EXPECT_EQ(Daytime::INVALID                , safe::from_ssm(86400));
}

TEST(Daytime, get_hms) {
  EXPECT_TRUE(check(safe::get_hms(Daytime::MIDNIGHT  ),  0,  0,  0.000));
  EXPECT_TRUE(check(safe::get_hms(from_ssm(    0.125)),  0,  0,  0.125));
  EXPECT_TRUE(check(safe::get_hms(from_ssm(    1.125)),  0,  0,  1.125));
  EXPECT_TRUE(check(safe::get_hms(from_ssm(   60.0  )),  0,  1,  0    ));
  EXPECT_TRUE(check(safe::get_hms(from_ssm(  600.0  )),  0, 10,  0    ));
  EXPECT_TRUE(check(safe::get_hms(from_ssm( 3600.0  )),  1,  0,  0    ));
  EXPECT_TRUE(check(safe::get_hms(from_ssm(43200.0  )), 12,  0,  0    ));
  EXPECT_TRUE(check(safe::get_hms(from_ssm(86399.875)), 23, 59, 59.875));

  EXPECT_TRUE (hms_is_valid(safe::get_hms(Daytime::MAX)));
  EXPECT_FALSE(hms_is_valid(safe::get_hms(Daytime::INVALID)));
  EXPECT_FALSE(hms_is_valid(safe::get_hms(Daytime::MISSING)));
}

TEST(Daytime, get_ssm) {
  EXPECT_EQ(    0           , safe::get_ssm(Daytime::MIDNIGHT));
  EXPECT_EQ(    0.0009765625, safe::get_ssm(from_hms( 0,  0,  0.0009765625)));
  EXPECT_EQ(    1           , safe::get_ssm(from_hms( 0,  0,  1)));
  EXPECT_EQ(   60           , safe::get_ssm(from_hms( 0,  1,  0)));
  EXPECT_EQ( 3600           , safe::get_ssm(from_hms( 1,  0,  0)));
  EXPECT_EQ(86399.9990234375, safe::get_ssm(from_hms(23, 59, 59.9990234375)));

  // SSM_INVALID is NaN; can't compare with ==.
  EXPECT_TRUE(std::isnan(safe::get_ssm(Daytime::INVALID)));
  EXPECT_TRUE(std::isnan(safe::get_ssm(Daytime::MISSING)));
}

TEST(Daytime, get_hour_minute_second) {
  EXPECT_EQ( 0    , safe::get_hour  (Daytime::MIDNIGHT));
  EXPECT_EQ( 0    , safe::get_minute(Daytime::MIDNIGHT));
  EXPECT_EQ( 0.0  , safe::get_second(Daytime::MIDNIGHT));

  auto d = Daytime::MIDNIGHT;
  EXPECT_EQ( 0    , safe::get_hour  (d));
  EXPECT_EQ( 0    , safe::get_minute(d));
  EXPECT_EQ( 0    , safe::get_second(d));

  d = from_hms(0, 0, 0.125);
  EXPECT_EQ( 0    , safe::get_hour  (d));
  EXPECT_EQ( 0    , safe::get_minute(d));
  EXPECT_EQ( 0.125, safe::get_second(d));

  d = from_hms(0, 1, 0);
  EXPECT_EQ( 0    , safe::get_hour  (d));
  EXPECT_EQ( 1    , safe::get_minute(d));
  EXPECT_EQ( 0    , safe::get_second(d));

  d = from_hms(1, 0, 0);
  EXPECT_EQ( 1    , safe::get_hour  (d));
  EXPECT_EQ( 0    , safe::get_minute(d));
  EXPECT_EQ( 0    , safe::get_second(d));

  d = from_hms(23, 59, 59.875);
  EXPECT_EQ(23    , safe::get_hour  (d));
  EXPECT_EQ(59    , safe::get_minute(d));
  EXPECT_EQ(59.875, safe::get_second(d));

  EXPECT_FALSE(hour_is_valid  (safe::get_hour  (Daytime::INVALID)));
  EXPECT_FALSE(minute_is_valid(safe::get_minute(Daytime::INVALID)));
  EXPECT_FALSE(second_is_valid(safe::get_second(Daytime::INVALID)));

  EXPECT_FALSE(hour_is_valid  (safe::get_hour  (Daytime::MISSING)));
  EXPECT_FALSE(minute_is_valid(safe::get_minute(Daytime::MISSING)));
  EXPECT_FALSE(second_is_valid(safe::get_second(Daytime::MISSING)));
}

