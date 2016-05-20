#include <string>

#include "gtest/gtest.h"
#include "cron/daytime.hh"
#include "cron/format.hh"

using namespace aslib;
using namespace cron;
using namespace cron::daytime;

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

TEST(Daytime, get_hms) {
  auto hms = Daytime::from_offset(0).get_hms();
  EXPECT_EQ(0, hms.hour);
  EXPECT_EQ(0, hms.minute);
  EXPECT_EQ(0, hms.second);

  hms = Daytime::from_ssm(60012.25).get_hms();
  EXPECT_EQ(16, hms.hour);
  EXPECT_EQ(40, hms.minute);
  EXPECT_EQ(12.25, hms.second);

  hms = Daytime(16, 40, 12.25).get_hms();
  EXPECT_EQ(16, hms.hour);
  EXPECT_EQ(40, hms.minute);
  EXPECT_EQ(12.25, hms.second);
}

TEST(Daytime, get_hms_invalid) {
  EXPECT_THROW(Daytime::INVALID.get_hms(), InvalidDaytimeError);
  EXPECT_THROW(Daytime::MISSING.get_hms(), InvalidDaytimeError);
}

TEST(Daytime, from_hms) {
  Daytime daytime = Daytime::from_hms(0, 0, 0);
  EXPECT_EQ(0u, daytime.get_offset());
  EXPECT_EQ(0u, daytime.get_ssm());

  daytime = Daytime::from_hms(16, 40, 12.25);
  EXPECT_EQ(8445973335552032768ull, daytime.get_offset());
  EXPECT_EQ(60012.25, daytime.get_ssm());
}

TEST(Daytime, from_hms_invalid) {
  EXPECT_THROW(Daytime(-1, 10, 10), InvalidDaytimeError);
  EXPECT_THROW(Daytime(24, 10, 10), InvalidDaytimeError);
  EXPECT_THROW(Daytime(12, -1, 10), InvalidDaytimeError);
  EXPECT_THROW(Daytime(12, 60, 10), InvalidDaytimeError);
  EXPECT_THROW(Daytime(12, 10, 60), InvalidDaytimeError);
  EXPECT_THROW(Daytime(12, 10, -1), InvalidDaytimeError);
}

TEST(Daytime, from_ssm) {
  EXPECT_EQ(Daytime( 0,  0,  0), Daytime::from_ssm(    0));
  EXPECT_EQ(Daytime( 0,  0,  1), Daytime::from_ssm(    1));
  EXPECT_EQ(Daytime( 0,  1,  0), Daytime::from_ssm(   60));
  EXPECT_EQ(Daytime( 1,  0,  0), Daytime::from_ssm( 3600));
  EXPECT_EQ(Daytime(23, 59, 59), Daytime::from_ssm(86399));
  EXPECT_EQ(Daytime( 0,  0,  0.001), Daytime::from_ssm(0.001));
}

TEST(Daytime, from_ssm_invalid) {
  EXPECT_THROW(Daytime::from_ssm(86400), InvalidDaytimeError);
  EXPECT_THROW(Daytime::from_ssm(-1), InvalidDaytimeError);
  EXPECT_THROW(Daytime::from_ssm(NaN), InvalidDaytimeError);
}

TEST(Daytime, get_ssm) {
  EXPECT_EQ(    0, Daytime( 0,  0,  0).get_ssm());
  EXPECT_EQ(    1, Daytime( 0,  0,  1).get_ssm());
  EXPECT_EQ(   60, Daytime( 0,  1,  0).get_ssm());
  EXPECT_EQ( 3600, Daytime( 1,  0,  0).get_ssm());
  EXPECT_EQ(86399, Daytime(23, 59, 59).get_ssm());

  EXPECT_NEAR(0.001, Daytime( 0,  0,  0.001).get_ssm(), 1e-8);
}

TEST(Daytime, get_ssm_invalid) {
  EXPECT_THROW(Daytime::INVALID.get_ssm(), InvalidDaytimeError);
  EXPECT_THROW(Daytime::MISSING.get_ssm(), InvalidDaytimeError);
}

TEST(Daytime, format) {
  Daytime const daytime = Daytime(15, 32, 10.0213);
  EXPECT_EQ("<15:32:10>",   format("<%H:%M:%S>", daytime));
  EXPECT_EQ("15 is 03 PM",  format("%H is %h %p", daytime));
}

TEST(Daytime, ostream) {
  auto const daytime = Daytime(15, 32, 0.7);
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
  auto const daytime = Daytime(15, 32, 0.7);
  EXPECT_EQ("15:32:01", to_string(daytime));
  EXPECT_EQ("3:32:00.70000 pm", to_string(DaytimeFormat("%0h:%M:%.5S %_p")(daytime)));
  EXPECT_EQ("3:32:00.70000 pm", (std::string) DaytimeFormat("%0h:%M:%.5S %_p")(daytime));
}

//------------------------------------------------------------------------------
// Class Daytime32
//------------------------------------------------------------------------------

// FIXME: Template tests?

TEST(Daytime32, get_hms) {
  auto hms = Daytime32::from_offset(0).get_hms();
  EXPECT_EQ(0, hms.hour);
  EXPECT_EQ(0, hms.minute);
  EXPECT_EQ(0, hms.second);

  hms = Daytime32::from_ssm(60012.25).get_hms();
  EXPECT_EQ(16, hms.hour);
  EXPECT_EQ(40, hms.minute);
  EXPECT_EQ(12.25, hms.second);
}

TEST(Daytime32, from_hms) {
  Daytime32 daytime = Daytime32(0, 0, 0);
  EXPECT_EQ(0u, daytime.get_offset());
  EXPECT_EQ(0.0, daytime.get_ssm());

  daytime = Daytime32(16, 40, 12.25);
  EXPECT_EQ(1966481408u, daytime.get_offset());
  EXPECT_EQ(60012.25, daytime.get_ssm());
}

TEST(Daytime32, from_hms_invalid) {
  EXPECT_THROW(Daytime32(24, 10, 10), InvalidDaytimeError);
  EXPECT_THROW(Daytime32(12, 60, 10), InvalidDaytimeError);
  EXPECT_THROW(Daytime32(12, 10, 60), InvalidDaytimeError);
}

