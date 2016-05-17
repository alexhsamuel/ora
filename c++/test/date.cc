#include <string>

#include "cron/date.hh"
#include "cron/ez.hh"
#include "cron/format.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::date;
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
  auto const ymd = get_ymd(Date(1973, 11, 2));
  EXPECT_EQ(1973, ymd.year);
  EXPECT_EQ(11, ymd.month);
  EXPECT_EQ(2, ymd.day);
}

TEST(Date, offsets) {
  EXPECT_EQ(Date::MIN, Date::from_offset(Date::MIN.get_offset()));
  EXPECT_EQ(Date::MIN, Date::from_datenum(Date::MIN.get_datenum()));

  EXPECT_EQ(Date::MAX, Date::from_offset(Date::MAX.get_offset()));
  EXPECT_EQ(Date::MAX, Date::from_datenum(Date::MAX.get_datenum()));

  Date const date1 = Date(1600, 2, 0);
  EXPECT_EQ(date1, Date::from_offset(date1.get_offset()));
  EXPECT_EQ(date1, Date::from_datenum(date1.get_datenum()));

  Date const date2 = Date(2000, 2, 0);
  EXPECT_EQ(date2, Date::from_offset(date2.get_offset()));
  EXPECT_EQ(date2, Date::from_datenum(date2.get_datenum()));
}

TEST(Date, shift) {
  Date const date0(1973, 11, 2);
  auto const date1 = date0 + 1;
  auto const ymd = get_ymd(date1);
  EXPECT_EQ(ymd.year, 1973);
  EXPECT_EQ(ymd.month, 11);
  EXPECT_EQ(ymd.day, 3);
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
  EXPECT_THROW(Date(1973, 11,  31), InvalidDateError);
  EXPECT_THROW(Date(1973, 10,  30), InvalidDateError);
  EXPECT_THROW(Date(1973,  1,  28), InvalidDateError);
  EXPECT_THROW(Date(1972,  1,  29), InvalidDateError);
  EXPECT_FALSE(Date(1972,  1,  28).is_invalid());
  EXPECT_THROW(Date(  -1,  0,   1), InvalidDateError);
  EXPECT_THROW(Date(   0,  0,   0), InvalidDateError);
  EXPECT_FALSE(Date(   1,  0,   0).is_invalid());
  EXPECT_FALSE(Date(1000,  0,   1).is_invalid());
  EXPECT_THROW(Date(1970, 12,   1), InvalidDateError);
  EXPECT_THROW(Date(64000, 0,   1), InvalidDateError);
}


TEST(Date, range_error) {
  EXPECT_THROW(Date::MIN  -       1, DateRangeError);
  EXPECT_THROW(Date::MIN  +      -1, DateRangeError);
  EXPECT_THROW(Date::MAX  +       1, DateRangeError);
  EXPECT_THROW(Date::MAX  +   10000, DateRangeError);
  EXPECT_THROW(Date::MAX  + 1000000, DateRangeError);
}

TEST(Date, invalid_ymd) {
  EXPECT_THROW(get_ymd(Date::INVALID), InvalidDateError);
  EXPECT_THROW(Date::INVALID.get_datenum(), InvalidDateError);
}

TEST(Date, missing_ymd) {
  EXPECT_THROW(get_ymd(Date::MISSING), InvalidDateError);
  EXPECT_THROW(Date::MISSING.get_datenum(), InvalidDateError);
}

TEST(Date, weekday) {
  EXPECT_EQ(MONDAY,             get_weekday(Date(   1,  0,  0)));
  EXPECT_EQ(MONDAY,             get_weekday(Date(1973, 11,  2)));
  EXPECT_EQ(THURSDAY,           get_weekday(Date(2016,  2, 23)));
  EXPECT_EQ(FRIDAY,             get_weekday(Date(9999, 11, 30)));

  EXPECT_THROW(get_weekday(Date::INVALID), InvalidDateError);
  EXPECT_THROW(get_weekday(Date::MISSING), InvalidDateError);
}

TEST(Date, conversions) {
  Date const date0 = 1973/DEC/ 3;
  EXPECT_EQ(Date16(1973, 11, 2), Date16(date0));

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

TEST(Date, from_iso_date) {
  EXPECT_EQ(from_iso_date<Date>("0001-01-01"),    1/JAN/ 1);
  EXPECT_EQ(from_iso_date<Date>("00010101"  ),    1/JAN/ 1);
  EXPECT_EQ(from_iso_date<Date>("1973-12-03"), 1973/DEC/ 3);
  EXPECT_EQ(from_iso_date<Date>("19731203"  ), 1973/DEC/ 3);
  EXPECT_EQ(from_iso_date<Date>("2016-02-29"), 2016/FEB/29);
  EXPECT_EQ(from_iso_date<Date>("20160229"  ), 2016/FEB/29);
  EXPECT_EQ(from_iso_date<Date>("9999-12-31"), 9999/DEC/31);
  EXPECT_EQ(from_iso_date<Date>("99991231"  ), 9999/DEC/31);
}

TEST(Date, from_iso_date_format_error) {
  EXPECT_THROW(from_iso_date<Date>(""), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("foobar"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("2000-01-1"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("2000-1-01"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("500-1-1"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("2000011"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("2000101"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("50011"), DateFormatError);
  EXPECT_THROW(from_iso_date<Date>("10000-01-01"), DateFormatError);
  EXPECT_THROW(
    from_iso_date<Date>("The quick brown fox jumped over the lazy dogs."), 
    DateFormatError);
}

TEST(Date, from_iso_date_invalid_error) {
  EXPECT_THROW(from_iso_date<Date>("2015-02-29"), InvalidDateError);
  EXPECT_THROW(from_iso_date<Date>("2015-03-32"), InvalidDateError);
  EXPECT_THROW(from_iso_date<Date>("2015-04-00"), InvalidDateError);
  EXPECT_THROW(from_iso_date<Date>("2015-13-01"), InvalidDateError);
}

TEST(Date16, from_iso_date) {
  EXPECT_EQ(from_iso_date<Date16>("1970-01-01"), Date16(1970, 0, 0));
  EXPECT_EQ(from_iso_date<Date16>("19700101"  ), Date16(1970, 0, 0));
  EXPECT_EQ(from_iso_date<Date16>("2149-06-04"), Date16(2149, 5, 3));
  EXPECT_EQ(from_iso_date<Date16>("21490604"  ), Date16(2149, 5, 3));
}

TEST(Date16, from_iso_date_range_error) {
  EXPECT_THROW(from_iso_date<Date16>("0001-01-01"), DateRangeError);
  EXPECT_THROW(from_iso_date<Date16>("9999-12-31"), DateRangeError);
  EXPECT_THROW(from_iso_date<Date16>("00010101"), DateRangeError);
  EXPECT_THROW(from_iso_date<Date16>("99991231"), DateRangeError);
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

  EXPECT_EQ(Date16(1970, 0, 0), Date16::MIN);
  EXPECT_EQ("1970-01-01", (string) DateFormat::ISO_CALENDAR_EXTENDED(Date16::MIN));
  EXPECT_EQ(Date16(2149, 5, 3), Date16::MAX);
  EXPECT_EQ("2149-06-04", (string) DateFormat::ISO_CALENDAR_EXTENDED(Date16::MAX));
}

TEST(Date16, is_valid) {
  EXPECT_TRUE (Date16::MIN.is_valid());
  EXPECT_TRUE (Date16::MAX.is_valid());
  EXPECT_FALSE(Date16::MISSING.is_valid());
  EXPECT_FALSE(Date16::INVALID.is_valid());
  EXPECT_TRUE (Date16(1970,  0, 0).is_valid());
  EXPECT_TRUE (Date16(1970,  0, 1).is_valid());
  EXPECT_TRUE (Date16(1973, 11, 2).is_valid());
}

TEST(Date16, from_ymd0) {
  Date16 const date{1970, 0, 0};
  EXPECT_EQ(0, date.get_offset());
  EXPECT_EQ(Date16(1970/JAN/ 1), date);
  EXPECT_TRUE(date.is(1970/JAN/ 1));
  EXPECT_TRUE(date.is({1970/JAN/ 1}));

  auto const ymd = get_ymd(date);
  EXPECT_EQ(1970,   ymd.year);
  EXPECT_EQ(0,      ymd.month);
  EXPECT_EQ(0,      ymd.day);
}

TEST(Date16, from_ymd1) {
  Date16 const date{1973, 11, 2};
  auto const ymd = get_ymd(date);
  EXPECT_EQ(1973,   ymd.year);
  EXPECT_EQ(11,     ymd.month);
  EXPECT_EQ(2,      ymd.day);
}

//------------------------------------------------------------------------------
// Easy literals.

TEST(MonthLiteral, basic) {
  EXPECT_EQ(Date(1973, 0, 2), 1973/JAN/3);
}

