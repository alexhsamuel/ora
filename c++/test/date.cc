#include <string>

#include "cron/date.hh"
#include "cron/ez.hh"
#include "cron/format.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::ez;

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
  EXPECT_EQ("9999-12-31", to_string(Date::MAX));
}

TEST(Date, from_ymd) {
  Date const date0 = Date::from_ymd(1973, 11, 2);
  DateParts const parts0 = date0.get_parts();
  EXPECT_EQ(1973, parts0.year);
  EXPECT_EQ(11, parts0.month);
  EXPECT_EQ(2, parts0.day);
}

TEST(Date, offsets) {
  EXPECT_EQ(Date::MIN, Date::from_offset(Date::MIN.get_offset()));
  EXPECT_EQ(Date::MIN, Date::from_datenum(Date::MIN.get_datenum()));

  EXPECT_EQ(Date::MAX, Date::from_offset(Date::MAX.get_offset()));
  EXPECT_EQ(Date::MAX, Date::from_datenum(Date::MAX.get_datenum()));

  Date const date1 = Date::from_ymd(1600, 2, 0);
  EXPECT_EQ(date1, Date::from_offset(date1.get_offset()));
  EXPECT_EQ(date1, Date::from_datenum(date1.get_datenum()));

  Date const date2 = Date::from_ymd(2000, 2, 0);
  EXPECT_EQ(date2, Date::from_offset(date2.get_offset()));
  EXPECT_EQ(date2, Date::from_datenum(date2.get_datenum()));
}

TEST(Date, shift) {
  Date const date0 = Date::from_ymd(1973, 11, 2);
  Date const date1 = date0 + 1;
  DateParts const parts1 = date1.get_parts();
  EXPECT_EQ(parts1.year, 1973);
  EXPECT_EQ(parts1.month, 11);
  EXPECT_EQ(parts1.day, 3);
}

TEST(Date, is_valid) {
  EXPECT_TRUE ((1973/DEC/ 3).is_valid());
  EXPECT_TRUE (Date::MIN.is_valid());
  EXPECT_TRUE (Date::MAX.is_valid());
  EXPECT_FALSE(Date::MISSING.is_valid());
  EXPECT_FALSE(Date::INVALID.is_valid());
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
  EXPECT_THROW(Date::from_datenum(DATENUM_INVALID).is_invalid(), InvalidDateError);
  EXPECT_THROW(Date::from_ymd(1973, 11,  31), InvalidDateError);
  EXPECT_THROW(Date::from_ymd(1973, 10,  30), InvalidDateError);
  EXPECT_THROW(Date::from_ymd(1973,  1,  28), InvalidDateError);
  EXPECT_THROW(Date::from_ymd(1972,  1,  29), InvalidDateError);
  EXPECT_FALSE(Date::from_ymd(1972,  1,  28).is_invalid());
  EXPECT_THROW(Date::from_ymd(  -1,  0,   1), InvalidDateError);
  EXPECT_THROW(Date::from_ymd(   0,  0,   0), InvalidDateError);
  EXPECT_FALSE(Date::from_ymd(   1,  0,   0).is_invalid());
  EXPECT_FALSE(Date::from_ymd(1000,  0,   1).is_invalid());
  EXPECT_THROW(Date::from_ymd(1970, 12,   1), InvalidDateError);
  EXPECT_THROW(Date::from_ymd(64000, 0,   1), InvalidDateError);
}


TEST(Date, range_error) {
  EXPECT_THROW(Date::MIN  -       1, DateRangeError);
  EXPECT_THROW(Date::MIN  +      -1, DateRangeError);
  EXPECT_THROW(Date::MAX  +       1, DateRangeError);
  EXPECT_THROW(Date::MAX  +   10000, DateRangeError);
  EXPECT_THROW(Date::MAX  + 1000000, DateRangeError);
}

TEST(Date, invalid_parts) {
  EXPECT_THROW(Date::INVALID.get_parts(), InvalidDateError);
  EXPECT_THROW(Date::INVALID.get_datenum(), InvalidDateError);
}

TEST(Date, missing_parts) {
  EXPECT_THROW(Date::MISSING.get_parts(), InvalidDateError);
  EXPECT_THROW(Date::MISSING.get_datenum(), InvalidDateError);
}

TEST(Date, weekday) {
  EXPECT_EQ(MONDAY,             Date::from_ymd(   1,  0,  0).get_weekday());
  EXPECT_EQ(MONDAY,             Date::from_ymd(1973, 11,  2).get_weekday());
  EXPECT_EQ(THURSDAY,           Date::from_ymd(2016,  2, 23).get_weekday());
  EXPECT_EQ(FRIDAY,             Date::from_ymd(9999, 11, 30).get_weekday());

  EXPECT_THROW(Date::INVALID.get_weekday(), InvalidDateError);
  EXPECT_THROW(Date::MISSING.get_weekday(), InvalidDateError);
}

TEST(Date, conversions) {
  Date const date0 = 1973/DEC/ 3;
  EXPECT_EQ(Date16::from_ymd(1973, 11, 2), Date16(date0));

  EXPECT_TRUE(Date16(Date::INVALID).is(Date16::INVALID));
  EXPECT_TRUE(Date16(Date::MISSING).is(Date16::MISSING));
  EXPECT_TRUE(Date(Date16::INVALID).is(Date::INVALID));
  EXPECT_TRUE(Date(Date16::MISSING).is(Date::MISSING));
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
// Class Date16
//------------------------------------------------------------------------------

TEST(Date16, default_ctor) {
  Date16 const date;
  EXPECT_TRUE(date.is_invalid());
}

TEST(Date16, range) {
  EXPECT_EQ(     0u, Date16::MIN.get_offset());
  EXPECT_EQ(719162u, Date16::MIN.get_datenum());

  EXPECT_EQ(Date16::from_ymd(1970, 0, 0), Date16::MIN);
  EXPECT_EQ("1970-01-01", (string) DateFormat::ISO_CALENDAR_EXTENDED(Date16::MIN));
  EXPECT_EQ(Date16::from_ymd(2149, 5, 3), Date16::MAX);
  EXPECT_EQ("2149-06-04", (string) DateFormat::ISO_CALENDAR_EXTENDED(Date16::MAX));
}

TEST(Date16, is_valid) {
  EXPECT_TRUE (Date16::MIN.is_valid());
  EXPECT_TRUE (Date16::MAX.is_valid());
  EXPECT_FALSE(Date16::MISSING.is_valid());
  EXPECT_FALSE(Date16::INVALID.is_valid());
  EXPECT_TRUE (Date16::from_ymd(1970,  0, 0).is_valid());
  EXPECT_TRUE (Date16::from_ymd(1970,  0, 1).is_valid());
  EXPECT_TRUE (Date16::from_ymd(1973, 11, 2).is_valid());
}

TEST(Date16, from_ymd) {
  Date16 const date0 = Date16::from_ymd(1970, 0, 0);
  EXPECT_EQ(0, date0.get_offset());
  EXPECT_EQ(Date16(1970/JAN/ 1), date0);
  EXPECT_TRUE(date0.is(1970/JAN/ 1));
  EXPECT_TRUE(date0.is({1970/JAN/ 1}));
  DateParts const parts0 = date0.get_parts();
  EXPECT_EQ(1970,   parts0.year);
  EXPECT_EQ(0,      parts0.month);
  EXPECT_EQ(0,      parts0.day);

  Date16 const date1 = Date16::from_ymd(1973, 11, 2);
  DateParts const parts1 = date1.get_parts();
  EXPECT_EQ(1973,   parts1.year);
  EXPECT_EQ(11,     parts1.month);
  EXPECT_EQ(2,      parts1.day);
}

//------------------------------------------------------------------------------
// Easy literals.

TEST(MonthLiteral, basic) {
  EXPECT_EQ(Date::from_ymd(1973, 0, 2), 1973/JAN/3);
}

