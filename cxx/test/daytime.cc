#include <string>

#include "ora.hh"
#include "gtest/gtest.h"

using namespace ora::lib;
using namespace ora;
using namespace ora::daytime;

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
  auto hms = get_hms(daytime::from_offset(0));
  EXPECT_EQ(0, hms.hour);
  EXPECT_EQ(0, hms.minute);
  EXPECT_EQ(0, hms.second);

  hms = get_hms(from_ssm<Daytime>(60012.25));
  EXPECT_EQ(16, hms.hour);
  EXPECT_EQ(40, hms.minute);
  EXPECT_EQ(12.25, hms.second);

  hms = get_hms(from_hms(16, 40, 12.25));
  EXPECT_EQ(16, hms.hour);
  EXPECT_EQ(40, hms.minute);
  EXPECT_EQ(12.25, hms.second);
}

TEST(Daytime, get_hms_invalid) {
  EXPECT_THROW(get_hms(Daytime::INVALID), InvalidDaytimeError);
  EXPECT_THROW(get_hms(Daytime::MISSING), InvalidDaytimeError);
}

TEST(Daytime, from_hms) {
  Daytime daytime = from_hms(0, 0, 0);
  EXPECT_EQ(0u, daytime.get_offset());
  EXPECT_EQ(0u, get_ssm(daytime));

  daytime = from_hms(16, 40, 12.25);
  EXPECT_EQ(8445973335552032768ull, daytime.get_offset());
  EXPECT_EQ(60012.25, get_ssm(daytime));
}

TEST(Daytime, from_hms_invalid) {
  EXPECT_THROW(from_hms(-1, 10, 10), InvalidDaytimeError);
  EXPECT_THROW(from_hms(24, 10, 10), InvalidDaytimeError);
  EXPECT_THROW(from_hms(12, -1, 10), InvalidDaytimeError);
  EXPECT_THROW(from_hms(12, 60, 10), InvalidDaytimeError);
  EXPECT_THROW(from_hms(12, 10, 60), InvalidDaytimeError);
  EXPECT_THROW(from_hms(12, 10, -1), InvalidDaytimeError);
}

TEST(Daytime, from_ssm) {
  EXPECT_EQ(from_hms( 0,  0,  0), from_ssm<Daytime>(    0));
  EXPECT_EQ(from_hms( 0,  0,  1), from_ssm<Daytime>(    1));
  EXPECT_EQ(from_hms( 0,  1,  0), from_ssm<Daytime>(   60));
  EXPECT_EQ(from_hms( 1,  0,  0), from_ssm<Daytime>( 3600));
  EXPECT_EQ(from_hms(23, 59, 59), from_ssm<Daytime>(86399));
  EXPECT_EQ(from_hms( 0,  0,  0.001), from_ssm<Daytime>(0.001));
}

TEST(Daytime, from_ssm_invalid) {
  EXPECT_THROW(from_ssm<Daytime>(86400), InvalidDaytimeError);
  EXPECT_THROW(from_ssm<Daytime>(-1), InvalidDaytimeError);
  EXPECT_THROW(from_ssm<Daytime>(NaN), InvalidDaytimeError);
}

TEST(Daytime, get_ssm) {
  EXPECT_EQ(    0, get_ssm(from_hms( 0,  0,  0)));
  EXPECT_EQ(    1, get_ssm(from_hms( 0,  0,  1)));
  EXPECT_EQ(   60, get_ssm(from_hms( 0,  1,  0)));
  EXPECT_EQ( 3600, get_ssm(from_hms( 1,  0,  0)));
  EXPECT_EQ(86399, get_ssm(from_hms(23, 59, 59)));

  EXPECT_NEAR(0.001, get_ssm(from_hms( 0,  0,  0.001)), 1e-8);
}

TEST(Daytime, get_ssm_invalid) {
  EXPECT_THROW(get_ssm(Daytime::INVALID), InvalidDaytimeError);
  EXPECT_THROW(get_ssm(Daytime::MISSING), InvalidDaytimeError);
}

TEST(Daytime, format) {
  Daytime const daytime = from_hms(15, 32, 10.0213);
  EXPECT_EQ("<15:32:10>",   format("<%H:%M:%S>", daytime));
  EXPECT_EQ("15 is 03 PM",  format("%H is %I %p", daytime));
}

TEST(Daytime, ostream) {
  auto const daytime = from_hms(15, 32, 0.75);
  {
    std::stringstream ss;
    ss << daytime;
    EXPECT_EQ("15:32:00", ss.str());
  }

  {
    std::stringstream ss;
    ss << DaytimeFormat::ISO_BASIC_MSEC(daytime);
    EXPECT_EQ("153200.750", ss.str());
  }
}

TEST(Daytime, to_string) {
  auto const daytime = from_hms(15, 32, 0.75);
  EXPECT_EQ("15:32:00", to_string(daytime));
  EXPECT_EQ("3:32:00.75000 pm", to_string(DaytimeFormat("%0I:%M:%.5S %_p")(daytime)));
  EXPECT_EQ("3:32:00.75000 pm", (std::string) DaytimeFormat("%0I:%M:%.5S %_p")(daytime));
}

//------------------------------------------------------------------------------
// Class Daytime32
//------------------------------------------------------------------------------

// FIXME: Template tests?

TEST(Daytime32, get_hms) {
  auto hms = get_hms(daytime::from_offset<Daytime32>(0));
  EXPECT_EQ(0, hms.hour);
  EXPECT_EQ(0, hms.minute);
  EXPECT_EQ(0, hms.second);

  hms = get_hms(from_ssm<Daytime32>(60012.25));
  EXPECT_EQ(16, hms.hour);
  EXPECT_EQ(40, hms.minute);
  EXPECT_EQ(12.25, hms.second);
}

TEST(Daytime32, from_hms) {
  Daytime32 daytime = from_hms<Daytime32>(0, 0, 0);
  EXPECT_EQ(0u, daytime.get_offset());
  EXPECT_EQ(0.0, get_ssm(daytime));

  daytime = from_hms(16, 40, 12.25);
  EXPECT_EQ(1966481408u, daytime.get_offset());
  EXPECT_EQ(60012.25, get_ssm(daytime));
}

TEST(Daytime32, from_hms_invalid) {
  EXPECT_THROW(from_hms<Daytime32>(24, 10, 10), InvalidDaytimeError);
  EXPECT_THROW(from_hms<Daytime32>(12, 60, 10), InvalidDaytimeError);
  EXPECT_THROW(from_hms<Daytime32>(12, 10, 60), InvalidDaytimeError);
}

TEST(Daytime, from_iso_daytime) {
  EXPECT_EQ(from_iso_daytime("0:00:00")        , from_hms( 0,  0,  0));
  EXPECT_EQ(from_iso_daytime("00:00:00")       , from_hms( 0,  0,  0));
  EXPECT_EQ(from_iso_daytime("00:00:00.")      , from_hms( 0,  0,  0));
  EXPECT_EQ(from_iso_daytime("00:00:00.0000")  , from_hms( 0,  0,  0));
  EXPECT_EQ(from_iso_daytime("00:00:00.5")     , from_hms( 0,  0,  0.5));
  EXPECT_EQ(from_iso_daytime("0:00:01")        , from_hms( 0,  0,  1));
  EXPECT_EQ(from_iso_daytime("00:01:00")       , from_hms( 0,  1,  0));
  EXPECT_EQ(from_iso_daytime("1:00:00")        , from_hms( 1,  0,  0));
  EXPECT_EQ(from_iso_daytime("01:00:00")       , from_hms( 1,  0,  0));
  EXPECT_EQ(from_iso_daytime("00:00:59.9375")  , from_hms( 0,  0, 59.9375));
  EXPECT_EQ(from_iso_daytime("00:59:00")       , from_hms( 0, 59,  0));
  EXPECT_EQ(from_iso_daytime("00:59:59.9375")  , from_hms( 0, 59, 59.9375));
  EXPECT_EQ(from_iso_daytime("23:00:00")       , from_hms(23,  0,  0));
  EXPECT_EQ(from_iso_daytime("23:59:59.9375")  , from_hms(23, 59, 59.9375));
}

TEST(Daytime, from_iso_daytime_format_error) {
  EXPECT_THROW(from_iso_daytime(""), DaytimeFormatError);
  EXPECT_THROW(from_iso_daytime("afternoon"), DaytimeFormatError);
  EXPECT_THROW(from_iso_daytime("0:0:0"), DaytimeFormatError);
  EXPECT_THROW(from_iso_daytime("12:3:00"), DaytimeFormatError);
  EXPECT_THROW(from_iso_daytime("12:30"), DaytimeFormatError);
  EXPECT_THROW(from_iso_daytime("00:00:60"), DaytimeFormatError);
  EXPECT_THROW(from_iso_daytime("00:60:00"), DaytimeFormatError);
  EXPECT_THROW(from_iso_daytime("24:00:00"), DaytimeFormatError);
}

TEST(Daytime, equal) {
  EXPECT_TRUE(Daytime::MIN == Daytime32::MIN);
  EXPECT_TRUE(Daytime::INVALID == Daytime32::INVALID);
  EXPECT_TRUE(Daytime::MISSING == Daytime32::MISSING);
  EXPECT_TRUE(from_hms(12,  0,  0) == from_hms<Daytime32>(12,  0,  0));
  EXPECT_TRUE(from_hms(23, 59, 59) == from_hms<Daytime32>(23, 59, 59));
}

