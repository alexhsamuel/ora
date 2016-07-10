#include "cron.hh"
#include "gtest/gtest.h"

using namespace cron;
using namespace cron::ez;

using std::string;

//------------------------------------------------------------------------------

TEST(DateFormat, date) {
  auto const date = 2013/AUG/7;
  EXPECT_EQ("2013-08-07", DateFormat("%Y-%m-%d")(date));
  EXPECT_EQ("8/7/13", DateFormat("%0m/%0d/%y")(date));
  EXPECT_THROW(DateFormat("%H:%M:%S")(date), TimeFormatError);
}

TEST(DateFormat, iso) {
  auto const date = 1985/APR/12;
  EXPECT_EQ("19850412",   DateFormat::ISO_CALENDAR_BASIC(date));
  EXPECT_EQ("1985-04-12", DateFormat::ISO_CALENDAR_EXTENDED(date));
  EXPECT_EQ("1985102",    DateFormat::ISO_ORDINAL_BASIC(date));
  EXPECT_EQ("1985-102",   DateFormat::ISO_ORDINAL_EXTENDED(date));
  EXPECT_EQ("1985W155",   DateFormat::ISO_WEEK_BASIC(date));
  EXPECT_EQ("1985-W15-5", DateFormat::ISO_WEEK_EXTENDED(date));
}

TEST(DateFormat, iso_special) {
  EXPECT_EQ("0001-01-01", DateFormat::ISO_CALENDAR_EXTENDED(Date::MIN));
  EXPECT_EQ("9999-12-31", DateFormat::ISO_CALENDAR_EXTENDED(Date::MAX));
  EXPECT_EQ("INVALID   ", DateFormat::ISO_CALENDAR_EXTENDED(Date::INVALID));
  EXPECT_EQ("MISSING   ", DateFormat::ISO_CALENDAR_EXTENDED(Date::MISSING));

  EXPECT_EQ("0001-001", DateFormat::ISO_ORDINAL_EXTENDED(Date::MIN));
  EXPECT_EQ("9999-365", DateFormat::ISO_ORDINAL_EXTENDED(Date::MAX));
  EXPECT_EQ("INVALID ", DateFormat::ISO_ORDINAL_EXTENDED(Date::INVALID));
  EXPECT_EQ("MISSING ", DateFormat::ISO_ORDINAL_EXTENDED(Date::MISSING));

  EXPECT_EQ("0001-W01-1", DateFormat::ISO_WEEK_EXTENDED(Date::MIN));
  EXPECT_EQ("9999-W52-5", DateFormat::ISO_WEEK_EXTENDED(Date::MAX));
  EXPECT_EQ("INVALID   ", DateFormat::ISO_WEEK_EXTENDED(Date::INVALID));
  EXPECT_EQ("MISSING   ", DateFormat::ISO_WEEK_EXTENDED(Date::MISSING));
}

TEST(DateFormat, iso_invalid) {
  EXPECT_EQ("INVALID ",   DateFormat::ISO_CALENDAR_BASIC(Date::INVALID));
  EXPECT_EQ("MISSING   ", DateFormat::ISO_CALENDAR_EXTENDED(Date::MISSING));
  EXPECT_EQ("INVALID",    DateFormat::ISO_ORDINAL_BASIC(Date::INVALID));
  EXPECT_EQ("INVALID ",   DateFormat::ISO_ORDINAL_EXTENDED(Date::INVALID));
  EXPECT_EQ("MISSING ",   DateFormat::ISO_WEEK_BASIC(Date::MISSING));
  EXPECT_EQ("INVALID   ", DateFormat::ISO_WEEK_EXTENDED(Date::INVALID));
}

TEST(Date, to_string1) {
  EXPECT_EQ("0001-01-01", to_string(Date::MIN));
  EXPECT_EQ("2016-06-05", to_string(2016/JUN/5));
  EXPECT_EQ("9999-12-31", to_string(Date::MAX));
  EXPECT_EQ("INVALID", to_string(Date::INVALID));
  EXPECT_EQ("MISSING", to_string(Date::MISSING));

  EXPECT_EQ("1970-01-01", to_string(Date16::MIN));
  EXPECT_EQ("2149-06-04", to_string(Date16::MAX));
  EXPECT_EQ("INVALID", to_string(Date16::INVALID));
  EXPECT_EQ("MISSING", to_string(Date16::MISSING));
}

