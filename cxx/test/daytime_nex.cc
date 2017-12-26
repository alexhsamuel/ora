#include "ora.hh"
#include "gtest/gtest.h"

using namespace ora;

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
  EXPECT_EQ(Daytime::MIN, daytime::nex::from_offset(0));
  EXPECT_EQ(daytime::from_offset(10000), daytime::nex::from_offset(10000));
  EXPECT_EQ(
    Daytime::INVALID, 
    daytime::nex::from_offset(Daytime::MAX.get_offset() + 1));
}

TEST(Daytime, from_daytick) {
  EXPECT_EQ(Daytime::MIN, daytime::nex::from_daytick(0));
  EXPECT_EQ(
    daytime::from_daytick(10000000), 
    daytime::nex::from_daytick(10000000));
  EXPECT_EQ(
    Daytime::INVALID, 
    daytime::nex::from_daytick(Daytime::MAX.get_daytick() + 1));
}

TEST(Daytime, from_hms) {
  EXPECT_EQ(Daytime::MIDNIGHT, daytime::nex::from_hms(0, 0));
  EXPECT_EQ(Daytime::MIDNIGHT, daytime::nex::from_hms(0, 0, 0));

  for (Hour h = 0; h < 24; ++h)
    for (Minute m = 0; m < 60; m += 5)
      for (Second s = 0; s < 60; s += 11)
        EXPECT_EQ(from_hms(h, m, s), daytime::nex::from_hms(h, m, s));
}

TEST(Daytime, from_hms_invalid) {
  EXPECT_EQ(Daytime::INVALID, daytime::nex::from_hms(-1,  0,  0));
  EXPECT_EQ(Daytime::INVALID, daytime::nex::from_hms(24,  0,  0));
  EXPECT_EQ(Daytime::INVALID, daytime::nex::from_hms( 0, -1,  0));
  EXPECT_EQ(Daytime::INVALID, daytime::nex::from_hms( 0, 60,  0));
  EXPECT_EQ(Daytime::INVALID, daytime::nex::from_hms( 0,  0, -1));
  EXPECT_EQ(Daytime::INVALID, daytime::nex::from_hms( 0,  0, 60));
}

TEST(Daytime, from_hms_struct) {
  struct HmsDaytime hms;
  EXPECT_EQ(Daytime::INVALID, daytime::nex::from_hms(hms));
  hms.hour = 0;
  EXPECT_EQ(Daytime::INVALID, daytime::nex::from_hms(hms));
  hms.minute = 0;
  EXPECT_EQ(Daytime::INVALID, daytime::nex::from_hms(hms));
  hms.second = 0;
  EXPECT_EQ(Daytime::MIDNIGHT, daytime::nex::from_hms(hms));

  hms = HmsDaytime{23, 59, 59.99609375};
  EXPECT_EQ(from_hms(23, 59, 59.99609375), daytime::nex::from_hms(hms));
}

TEST(Daytime, from_ssm) {
  EXPECT_EQ(Daytime::INVALID                , daytime::nex::from_ssm(-0.01));
  EXPECT_EQ(Daytime::MIDNIGHT               , daytime::nex::from_ssm(0));
  EXPECT_EQ(from_hms(23, 59, 59)            , daytime::nex::from_ssm(86399));
  EXPECT_EQ(from_hms(23, 59, 59.99609375)   , daytime::nex::from_ssm(86399.99609375));
  EXPECT_EQ(Daytime::INVALID                , daytime::nex::from_ssm(86400));
}

TEST(Daytime, get_hms) {
  EXPECT_TRUE(check(daytime::nex::get_hms(Daytime::MIDNIGHT  ),  0,  0,  0.000));
  EXPECT_TRUE(check(daytime::nex::get_hms(from_ssm(    0.125)),  0,  0,  0.125));
  EXPECT_TRUE(check(daytime::nex::get_hms(from_ssm(    1.125)),  0,  0,  1.125));
  EXPECT_TRUE(check(daytime::nex::get_hms(from_ssm(   60.0  )),  0,  1,  0    ));
  EXPECT_TRUE(check(daytime::nex::get_hms(from_ssm(  600.0  )),  0, 10,  0    ));
  EXPECT_TRUE(check(daytime::nex::get_hms(from_ssm( 3600.0  )),  1,  0,  0    ));
  EXPECT_TRUE(check(daytime::nex::get_hms(from_ssm(43200.0  )), 12,  0,  0    ));
  EXPECT_TRUE(check(daytime::nex::get_hms(from_ssm(86399.875)), 23, 59, 59.875));

  EXPECT_TRUE (hms_is_valid(daytime::nex::get_hms(Daytime::MAX)));
  EXPECT_FALSE(hms_is_valid(daytime::nex::get_hms(Daytime::INVALID)));
  EXPECT_FALSE(hms_is_valid(daytime::nex::get_hms(Daytime::MISSING)));
}

TEST(Daytime, get_ssm) {
  EXPECT_EQ(    0           , daytime::nex::get_ssm(Daytime::MIDNIGHT));
  EXPECT_EQ(    0.0009765625, daytime::nex::get_ssm(from_hms( 0,  0,  0.0009765625)));
  EXPECT_EQ(    1           , daytime::nex::get_ssm(from_hms( 0,  0,  1)));
  EXPECT_EQ(   60           , daytime::nex::get_ssm(from_hms( 0,  1,  0)));
  EXPECT_EQ( 3600           , daytime::nex::get_ssm(from_hms( 1,  0,  0)));
  EXPECT_EQ(86399.9990234375, daytime::nex::get_ssm(from_hms(23, 59, 59.9990234375)));

  // SSM_INVALID is NaN; can't compare with ==.
  EXPECT_TRUE(std::isnan(daytime::nex::get_ssm(Daytime::INVALID)));
  EXPECT_TRUE(std::isnan(daytime::nex::get_ssm(Daytime::MISSING)));
}

TEST(Daytime, get_hour_minute_second) {
  EXPECT_EQ( 0    , daytime::nex::get_hour  (Daytime::MIDNIGHT));
  EXPECT_EQ( 0    , daytime::nex::get_minute(Daytime::MIDNIGHT));
  EXPECT_EQ( 0.0  , daytime::nex::get_second(Daytime::MIDNIGHT));

  auto d = Daytime::MIDNIGHT;
  EXPECT_EQ( 0    , daytime::nex::get_hour  (d));
  EXPECT_EQ( 0    , daytime::nex::get_minute(d));
  EXPECT_EQ( 0    , daytime::nex::get_second(d));

  d = from_hms(0, 0, 0.125);
  EXPECT_EQ( 0    , daytime::nex::get_hour  (d));
  EXPECT_EQ( 0    , daytime::nex::get_minute(d));
  EXPECT_EQ( 0.125, daytime::nex::get_second(d));

  d = from_hms(0, 1, 0);
  EXPECT_EQ( 0    , daytime::nex::get_hour  (d));
  EXPECT_EQ( 1    , daytime::nex::get_minute(d));
  EXPECT_EQ( 0    , daytime::nex::get_second(d));

  d = from_hms(1, 0, 0);
  EXPECT_EQ( 1    , daytime::nex::get_hour  (d));
  EXPECT_EQ( 0    , daytime::nex::get_minute(d));
  EXPECT_EQ( 0    , daytime::nex::get_second(d));

  d = from_hms(23, 59, 59.875);
  EXPECT_EQ(23    , daytime::nex::get_hour  (d));
  EXPECT_EQ(59    , daytime::nex::get_minute(d));
  EXPECT_EQ(59.875, daytime::nex::get_second(d));

  EXPECT_FALSE(hour_is_valid  (daytime::nex::get_hour  (Daytime::INVALID)));
  EXPECT_FALSE(minute_is_valid(daytime::nex::get_minute(Daytime::INVALID)));
  EXPECT_FALSE(second_is_valid(daytime::nex::get_second(Daytime::INVALID)));

  EXPECT_FALSE(hour_is_valid  (daytime::nex::get_hour  (Daytime::MISSING)));
  EXPECT_FALSE(minute_is_valid(daytime::nex::get_minute(Daytime::MISSING)));
  EXPECT_FALSE(second_is_valid(daytime::nex::get_second(Daytime::MISSING)));
}

TEST(Daytime, seconds_after) {
  EXPECT_EQ(Daytime::INVALID, daytime::nex::seconds_after(Daytime::INVALID, 0));
  EXPECT_EQ(Daytime::INVALID, daytime::nex::seconds_after(Daytime::MISSING, 1));

  EXPECT_EQ(from_hms( 0,  1,  5), daytime::nex::seconds_after(from_hms( 0,  0, 55), 10));
  EXPECT_EQ(from_hms(12,  0,  5), daytime::nex::seconds_after(from_hms(11, 59, 55), 10));

  for (double sec = 0; sec < 86400; sec += 9.875) 
    EXPECT_EQ(from_ssm(sec), daytime::nex::seconds_after(Daytime::MIDNIGHT, sec));

  EXPECT_EQ(Daytime::MIDNIGHT, daytime::nex::seconds_after(Daytime::MIDNIGHT, 86400));
  EXPECT_EQ(from_hms(0, 0, 0.125), daytime::nex::seconds_after(from_hms(23, 59, 59.5), 0.625));
}

TEST(Daytime, seconds_before) {
  EXPECT_EQ(Daytime::INVALID, daytime::nex::seconds_before(Daytime::INVALID, 0));
  EXPECT_EQ(Daytime::INVALID, daytime::nex::seconds_before(Daytime::MISSING, 1));

  EXPECT_EQ(from_hms( 0,  0, 55), daytime::nex::seconds_before(from_hms( 0,  1,  5), 10));
  EXPECT_EQ(from_hms(11, 59, 55), daytime::nex::seconds_before(from_hms(12,  0,  5), 10));

  for (double sec = 0; sec < 86400; sec += 9.875) 
    EXPECT_EQ(Daytime::MIDNIGHT, daytime::nex::seconds_before(from_ssm(sec), sec));

  EXPECT_EQ(from_hms(23, 59, 59.875), daytime::nex::seconds_before(Daytime::MIDNIGHT, 0.125));
  EXPECT_EQ(from_hms(23, 59, 59.875), daytime::nex::seconds_before(Daytime::MIDNIGHT, 86400.125));
}

TEST(Daytime, seconds_between) {
  EXPECT_TRUE(std::isnan(daytime::nex::seconds_between(Daytime::INVALID, Daytime::MIN)));
  EXPECT_TRUE(std::isnan(daytime::nex::seconds_between(Daytime::MISSING, Daytime::MIN)));
  EXPECT_TRUE(std::isnan(daytime::nex::seconds_between(Daytime::MIN, Daytime::INVALID)));
  EXPECT_TRUE(std::isnan(daytime::nex::seconds_between(Daytime::MIN, Daytime::MISSING)));

  auto const a0 = Daytime::MIDNIGHT;
  auto const a1 = from_hms( 0,  0,  0.25);
  auto const a2 = from_hms(23,  0,  0);
  EXPECT_EQ(     0.0 , daytime::nex::seconds_between(a0, a0));
  EXPECT_EQ(     0.0 , daytime::nex::seconds_between(a1, a1));
  EXPECT_EQ(     0.0 , daytime::nex::seconds_between(a2, a2));
  EXPECT_EQ(     0.25, daytime::nex::seconds_between(a0, a1));
  EXPECT_EQ(    -0.25, daytime::nex::seconds_between(a1, a0));
  EXPECT_EQ( 82800.0 , daytime::nex::seconds_between(a0, a2));
  EXPECT_EQ(-82800.0 , daytime::nex::seconds_between(a2, a0));
  EXPECT_EQ( 82799.75, daytime::nex::seconds_between(a1, a2));
  EXPECT_EQ(-82799.75, daytime::nex::seconds_between(a2, a1));
}

