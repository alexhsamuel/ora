#include <string>

#include "cron/date.hh"
#include "cron/ez.hh"
#include "cron/format.hh"
#include "gtest/gtest.h"

using namespace alxs;
using namespace alxs::cron;
using namespace alxs::cron::ez;

using std::string;

//------------------------------------------------------------------------------
// Class Date
//------------------------------------------------------------------------------

TEST(Date, default_ctor) {
  Date const date;
  EXPECT_TRUE(date.is_invalid());
}

TEST(Date, range) {
  EXPECT_EQ("0001-01-01", to_string(Date::MIN));
  EXPECT_EQ("9999-12-31", to_string(Date::LAST));
  EXPECT_EQ("INVALID   ", to_string(Date::MAX));
}

TEST(Date, from_ymd) {
  Date const date0 = Date::from_parts(1973, 11, 2);
  DateParts const parts0 = date0.get_parts();
  EXPECT_EQ(1973, parts0.year);
  EXPECT_EQ(11, parts0.month);
  EXPECT_EQ(2, parts0.day);
}

TEST(Date, offsets) {
  EXPECT_EQ(Date::MIN, Date::from_offset(Date::MIN.get_offset()));
  EXPECT_EQ(Date::MIN, Date::from_datenum(Date::MIN.get_datenum()));

  EXPECT_EQ(Date::LAST, Date::from_offset(Date::LAST.get_offset()));
  EXPECT_EQ(Date::LAST, Date::from_datenum(Date::LAST.get_datenum()));

  Date const date1 = Date::from_parts(1600, 2, 0);
  EXPECT_EQ(date1, Date::from_offset(date1.get_offset()));
  EXPECT_EQ(date1, Date::from_datenum(date1.get_datenum()));

  Date const date2 = Date::from_parts(2000, 2, 0);
  EXPECT_EQ(date2, Date::from_offset(date2.get_offset()));
  EXPECT_EQ(date2, Date::from_datenum(date2.get_datenum()));
}

TEST(Date, shift) {
  Date const date0 = Date::from_parts(1973, 11, 2);
  Date const date1 = shift(date0, 1);
  DateParts const parts1 = date1.get_parts();
  EXPECT_EQ(parts1.year, 1973);
  EXPECT_EQ(parts1.month, 11);
  EXPECT_EQ(parts1.day, 3);
}

TEST(Date, is_valid) {
  EXPECT_TRUE ((1973/DEC/ 3).is_valid());
  EXPECT_TRUE (Date::MIN.is_valid());
  EXPECT_TRUE (Date::LAST.is_valid());
  EXPECT_FALSE(Date::MAX.is_valid());
  EXPECT_FALSE(Date::INVALID.is_valid());
  EXPECT_FALSE(Date::MISSING.is_valid());
}

TEST(Date, is) {
  Date const date0 = 1973/DEC/ 3;

  EXPECT_TRUE (date0.is(date0));
  EXPECT_FALSE(date0.is(Date::MISSING));
  EXPECT_FALSE(date0.is_missing());
  EXPECT_FALSE(date0.is(Date::INVALID));
  EXPECT_FALSE(date0.is_invalid());

  EXPECT_FALSE(Date::INVALID.is(date0));
  EXPECT_TRUE (Date::MISSING.is(Date::MISSING));
  EXPECT_TRUE (Date::MISSING.is_missing());
  EXPECT_FALSE(Date::MISSING.is(Date::INVALID));
  EXPECT_FALSE(Date::MISSING.is_invalid());

  EXPECT_FALSE(Date::INVALID.is(date0));
  EXPECT_FALSE(Date::INVALID.is(Date::MISSING));
  EXPECT_FALSE(Date::INVALID.is_missing());
  EXPECT_TRUE (Date::INVALID.is(Date::INVALID));
  EXPECT_TRUE (Date::INVALID.is_invalid());
}

TEST(Date, invalid) {
  EXPECT_TRUE (Date::from_datenum(DATENUM_INVALID).is_invalid());
  EXPECT_TRUE (Date::from_parts(1973, 11,  31).is_invalid());
  EXPECT_TRUE (Date::from_parts(1973, 10,  30).is_invalid());
  EXPECT_TRUE (Date::from_parts(1973,  1,  28).is_invalid());
  EXPECT_TRUE (Date::from_parts(1972,  1,  29).is_invalid());
  EXPECT_FALSE(Date::from_parts(1972,  1,  28).is_invalid());
  EXPECT_TRUE (Date::from_parts(  -1,  0,   1).is_invalid());
  EXPECT_TRUE (Date::from_parts(   0,  0,   0).is_invalid());
  EXPECT_FALSE(Date::from_parts(   1,  0,   0).is_invalid());
  EXPECT_FALSE(Date::from_parts(1000,  0,   1).is_invalid());
  EXPECT_TRUE (Date::from_parts(1970, 12,   1).is_invalid());
  EXPECT_TRUE (Date::from_parts(64000, 0,   1).is_invalid());
  EXPECT_TRUE (shift(Date::MIN,       -1).is_invalid());
  EXPECT_TRUE (shift(Date::LAST,       1).is_invalid());
  EXPECT_TRUE (shift(Date::LAST,   10000).is_invalid());
  EXPECT_TRUE (shift(Date::LAST, 1000000).is_invalid());
}

TEST(Date, invalid_parts) {
  DateParts const parts = Date::INVALID.get_parts();
  EXPECT_EQ(DAY_INVALID,        parts.day);
  EXPECT_EQ(MONTH_INVALID,      parts.month);
  EXPECT_EQ(YEAR_INVALID,       parts.year);
  EXPECT_EQ(WEEKDAY_INVALID,    parts.weekday);
  EXPECT_EQ(WEEK_INVALID,       parts.week);
  EXPECT_EQ(YEAR_INVALID,       parts.week_year);
}

TEST(Date, missing_parts) {
  DateParts const parts = Date::MISSING.get_parts();
  EXPECT_EQ(DAY_INVALID,        parts.day);
  EXPECT_EQ(MONTH_INVALID,      parts.month);
  EXPECT_EQ(YEAR_INVALID,       parts.year);
  EXPECT_EQ(WEEKDAY_INVALID,    parts.weekday);
  EXPECT_EQ(WEEK_INVALID,       parts.week);
  EXPECT_EQ(YEAR_INVALID,       parts.week_year);
}

TEST(Date, weekday) {
  EXPECT_EQ(WEEKDAY_INVALID,    Date::INVALID.get_weekday());
  EXPECT_EQ(MONDAY,             Date::from_parts(1973, 11, 2).get_weekday());
}

TEST(Date, conversions) {
  Date const date0 = 1973/DEC/ 3;
  EXPECT_EQ(SmallDate::from_parts(1973, 11, 2), SmallDate(date0));

  EXPECT_TRUE(SmallDate(Date::INVALID).is(SmallDate::INVALID));
  EXPECT_TRUE(SmallDate(Date::MISSING).is(SmallDate::MISSING));
  EXPECT_TRUE(Date(SmallDate::INVALID).is(Date::INVALID));
  EXPECT_TRUE(Date(SmallDate::MISSING).is(Date::MISSING));
}

TEST(Date, ostream) {
  Date const date = 1973/DEC/3;
  {
    std::stringstream ss;
    ss << date;
    EXPECT_EQ("1973-12-03", ss.str());
  }

  {
    DateFormat const format = "%0m/%0d/%y";
    std::stringstream ss;
    ss << format(date);
    EXPECT_EQ("12/3/73", ss.str());
  }

}

TEST(Date, to_string) {
  Date const date = 1973/DEC/3;
  EXPECT_EQ("1973-12-03", to_string(date));

  DateFormat const format = "%~^b %0d, %Y";
  EXPECT_EQ("DEC 3, 1973", to_string(format(date)));

  EXPECT_EQ("1973-12-03", to_string(date));
  EXPECT_EQ("1973 Dec  3", (std::string) DateFormat("%Y %~b %# d")(date));
  EXPECT_EQ("INVALID   ", to_string(Date::INVALID));
  EXPECT_EQ("MISSING   ", to_string(Date::MISSING));
}

//------------------------------------------------------------------------------
// Class SafeDate
//------------------------------------------------------------------------------

TEST(SafeDate, default_ctor) {
  SafeDate const date;
  EXPECT_EQ(SafeDate::MIN, date);
}

TEST(SafeDate, range) {
  EXPECT_EQ("0001-01-01", to_string(SafeDate::MIN));
  EXPECT_EQ("9999-12-31", to_string(SafeDate::LAST));
  EXPECT_EQ("INVALID   ", to_string(SafeDate::MAX));
}

TEST(SafeDate, invalid) {
  EXPECT_THROW   (SafeDate::from_datenum(DATENUM_INVALID), ValueError);
  EXPECT_THROW   (SafeDate::from_parts(1973, 11, 31), ValueError);
  EXPECT_THROW   (SafeDate::from_parts(1973, 10, 30), ValueError);
  EXPECT_THROW   (SafeDate::from_parts(1973,  1, 28), ValueError);
  EXPECT_THROW   (SafeDate::from_parts(1972,  1, 29), ValueError);
  EXPECT_NO_THROW(SafeDate::from_parts(1972,  1, 28));
  EXPECT_THROW   (SafeDate::from_parts(  -1,  0,   1), ValueError);
  EXPECT_THROW   (SafeDate::from_parts(   0,  0,   1), ValueError);
  EXPECT_THROW   (SafeDate::from_parts(   0, -1,   1), ValueError);
  EXPECT_THROW   (SafeDate::from_parts(1970, 12,   1), ValueError);
  EXPECT_THROW   (SafeDate::from_parts(64000, 0,   1), ValueError);
  EXPECT_THROW   (SafeDate date(Date::INVALID), ValueError);
  EXPECT_THROW   (SafeDate date(Date::MISSING), ValueError);
  EXPECT_THROW   (shift(SafeDate::MIN, -1),                ValueError);
  EXPECT_THROW   (shift(SafeDate::LAST, 1),                ValueError);
  EXPECT_THROW   (shift(SafeDate::LAST, 10000),            ValueError);
  EXPECT_THROW   (shift(SafeDate::LAST, 1000000),          ValueError);
}

//------------------------------------------------------------------------------
// Class SmallDate
//------------------------------------------------------------------------------

TEST(SmallDate, default_ctor) {
  SmallDate const date;
  EXPECT_TRUE(date.is_invalid());
}

TEST(SmallDate, range) {
  EXPECT_EQ(     0u, SmallDate::MIN.get_offset());
  EXPECT_EQ(719162u, SmallDate::MIN.get_datenum());

  EXPECT_EQ(SmallDate::from_parts(1970, 0, 0), SmallDate::MIN);
  EXPECT_EQ("1970-01-01", (string) DateFormat::ISO_CALENDAR_EXTENDED(SmallDate::MIN));
  EXPECT_EQ(SmallDate::from_parts(2149, 5, 3), SmallDate::LAST);
  EXPECT_EQ("2149-06-04", (string) DateFormat::ISO_CALENDAR_EXTENDED(SmallDate::LAST));
}

TEST(SmallDate, is_valid) {
  EXPECT_TRUE (SmallDate::MIN.is_valid());
  EXPECT_TRUE (SmallDate::LAST.is_valid());
  EXPECT_FALSE(SmallDate::MAX.is_valid());
  EXPECT_FALSE(SmallDate::INVALID.is_valid());
  EXPECT_FALSE(SmallDate::MISSING.is_valid());
  EXPECT_TRUE (SmallDate::from_parts(1970,  0, 0).is_valid());
  EXPECT_TRUE (SmallDate::from_parts(1970,  0, 1).is_valid());
  EXPECT_TRUE (SmallDate::from_parts(1973, 11, 2).is_valid());
}

TEST(SmallDate, from_ymd) {
  SmallDate const date0 = SmallDate::from_parts(1970, 0, 0);
  EXPECT_EQ(0, date0.get_offset());
  EXPECT_EQ(SmallDate(1970/JAN/ 1), date0);
  EXPECT_TRUE(date0.is(1970/JAN/ 1));
  EXPECT_TRUE(date0.is({1970/JAN/ 1}));
  DateParts const parts0 = date0.get_parts();
  EXPECT_EQ(1970,   parts0.year);
  EXPECT_EQ(0,      parts0.month);
  EXPECT_EQ(0,      parts0.day);

  SmallDate const date1 = SmallDate::from_parts(1973, 11, 2);
  DateParts const parts1 = date1.get_parts();
  EXPECT_EQ(1973,   parts1.year);
  EXPECT_EQ(11,     parts1.month);
  EXPECT_EQ(2,      parts1.day);
}

//------------------------------------------------------------------------------
// Easy literals.

TEST(MonthLiteral, basic) {
  using namespace alxs::cron::ez;
  EXPECT_EQ(Date::from_parts(1973, 0, 2), 1973/JAN/3);
}

