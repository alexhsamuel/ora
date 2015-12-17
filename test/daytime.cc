#include <string>

#include "gtest/gtest.h"
#include "cron/daytime.hh"
#include "cron/format.hh"

using namespace alxs;
using namespace alxs::cron;

using std::string;

double constexpr NaN = std::numeric_limits<double>::quiet_NaN();

//------------------------------------------------------------------------------
// Class Daytime.

inline std::string
format(
  char const* pattern,
  Daytime daytime)
{
  return (string) DaytimeFormat(pattern)(daytime);
}


TEST(Daytime, get_parts) {
  DaytimeParts parts = Daytime::from_offset(0).get_parts();
  EXPECT_EQ(0, parts.hour);
  EXPECT_EQ(0, parts.minute);
  EXPECT_EQ(0, parts.second);

  parts = Daytime::from_ssm(60012.25).get_parts();
  EXPECT_EQ(16, parts.hour);
  EXPECT_EQ(40, parts.minute);
  EXPECT_EQ(12.25, parts.second);

  parts = Daytime(16, 40, 12.25).get_parts();
  EXPECT_EQ(16, parts.hour);
  EXPECT_EQ(40, parts.minute);
  EXPECT_EQ(12.25, parts.second);

  parts = Daytime::INVALID.get_parts();
  EXPECT_FALSE(hour_is_valid(parts.hour));
  EXPECT_FALSE(minute_is_valid(parts.minute));
  EXPECT_FALSE(second_is_valid(parts.second));
}

TEST(Daytime, from_hms) {
  Daytime daytime = Daytime(0, 0, 0);
  EXPECT_EQ(0, daytime.get_offset());
  EXPECT_EQ(0, daytime.get_ssm());

  daytime = Daytime(16, 40, 12.25);
  EXPECT_EQ(8445973335552032768ll, daytime.get_offset());
  EXPECT_EQ(60012.25, daytime.get_ssm());

  EXPECT_TRUE(Daytime(-1, 10, 10).is_invalid());
  EXPECT_TRUE(Daytime(24, 10, 10).is_invalid());
  EXPECT_TRUE(Daytime(12, -1, 10).is_invalid());
  EXPECT_TRUE(Daytime(12, 60, 10).is_invalid());
  EXPECT_TRUE(Daytime(12, 10, 60).is_invalid());
  EXPECT_TRUE(Daytime(12, 10, -1).is_invalid());
}

TEST(Daytime, from_ssm) {
  EXPECT_EQ(Daytime( 0,  0,  0), Daytime::from_ssm(    0));
  EXPECT_EQ(Daytime( 0,  0,  1), Daytime::from_ssm(    1));
  EXPECT_EQ(Daytime( 0,  1,  0), Daytime::from_ssm(   60));
  EXPECT_EQ(Daytime( 1,  0,  0), Daytime::from_ssm( 3600));
  EXPECT_EQ(Daytime(23, 59, 59), Daytime::from_ssm(86399));
  EXPECT_EQ(Daytime( 0,  0,  0.001), Daytime::from_ssm(0.001));

  EXPECT_TRUE(Daytime::from_ssm(86400).is_invalid());
  EXPECT_TRUE(Daytime::from_ssm(-1).is_invalid());
  EXPECT_TRUE(Daytime::from_ssm(NaN).is_invalid());
}

TEST(Daytime, get_ssm) {
  EXPECT_EQ(    0, Daytime( 0,  0,  0).get_ssm());
  EXPECT_EQ(    1, Daytime( 0,  0,  1).get_ssm());
  EXPECT_EQ(   60, Daytime( 0,  1,  0).get_ssm());
  EXPECT_EQ( 3600, Daytime( 1,  0,  0).get_ssm());
  EXPECT_EQ(86399, Daytime(23, 59, 59).get_ssm());

  EXPECT_NEAR(0.001, Daytime( 0,  0,  0.001).get_ssm(), 1e-8);

  EXPECT_FALSE(ssm_is_valid(Daytime::INVALID.get_ssm()));
  EXPECT_FALSE(ssm_is_valid(Daytime::MISSING.get_ssm()));
}

TEST(Daytime, format) {
  Daytime const daytime = Daytime(15, 32, 10.0213);
  EXPECT_EQ("<15:32:10>",   format("<%H:%M:%S>", daytime));
  EXPECT_EQ("15 is 03 PM",  format("%H is %h %p", daytime));
}

TEST(Daytime, ostream) {
  Daytime const daytime(15, 32, 0.7);
  {
    std::stringstream ss;
    ss << daytime;
    EXPECT_EQ("15:32:01", ss.str());
  }

  {
    std::stringstream ss;
    ss << DaytimeFormat::ISO_BASIC_MSEC(daytime);
    EXPECT_EQ("153200.700", ss.str());
  }
}

TEST(Daytime, to_string) {
  Daytime const daytime(15, 32, 0.7);
  EXPECT_EQ("15:32:01", to_string(daytime));
  EXPECT_EQ("3:32:00.70000 pm", to_string(DaytimeFormat("%0h:%M:%.5S %_p")(daytime)));
  EXPECT_EQ("3:32:00.70000 pm", (std::string) DaytimeFormat("%0h:%M:%.5S %_p")(daytime));
}

//------------------------------------------------------------------------------
// Class SmallDaytime
//------------------------------------------------------------------------------

// FIXME: Template tests?

TEST(SmallDaytime, get_parts) {
  auto parts = SmallDaytime::from_offset(0).get_parts();
  EXPECT_EQ(0, parts.hour);
  EXPECT_EQ(0, parts.minute);
  EXPECT_EQ(0, parts.second);

  parts = SmallDaytime::from_ssm(60012.25).get_parts();
  EXPECT_EQ(16, parts.hour);
  EXPECT_EQ(40, parts.minute);
  EXPECT_EQ(12.25, parts.second);
}

TEST(SmallDaytime, from_hms) {
  SmallDaytime daytime = SmallDaytime(0, 0, 0);
  EXPECT_EQ(0, daytime.get_offset());
  EXPECT_EQ(0.0, daytime.get_ssm());

  daytime = SmallDaytime(16, 40, 12.25);
  EXPECT_EQ(1966481408, daytime.get_offset());
  EXPECT_EQ(60012.25, daytime.get_ssm());

  EXPECT_TRUE(SmallDaytime(24, 10, 10).is_invalid());
  EXPECT_TRUE(SmallDaytime(12, 60, 10).is_invalid());
  EXPECT_TRUE(SmallDaytime(12, 10, 60).is_invalid());
}

