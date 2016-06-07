#include "cron/ez.hh"
#include "cron/format.hh"
#include "cron/localization.hh"
#include "gtest/gtest.h"

using namespace aslib;
using namespace cron;
using namespace cron::date;
using namespace cron::daytime;
using namespace cron::ez;
using namespace cron::time;

using std::string;

//------------------------------------------------------------------------------
// Class TimeFormat
//------------------------------------------------------------------------------

TEST(TimeFormat, basic) {
  Time const time = Time::from_offset(4262126704878682112l);
  auto const tz = get_time_zone("US/Eastern");
  EXPECT_EQ("2013-07-28",       TimeFormat("%Y-%m-%d")(time, *tz));
  EXPECT_EQ("15:37:38",         TimeFormat("%H:%M:%S")(time, *tz));
  EXPECT_THROW(TimeFormat("foo %c")(time), TimeFormatError);
}

TEST(TimeFormat, invalid) {
  // FIXME
}

TEST(TimeFormat, all) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/28, Daytime(15, 37, 38.0), *tz);
  EXPECT_EQ("July (Jul)",       TimeFormat("%b (%~b)")(time, *tz));
  EXPECT_THROW(TimeFormat("%c")(time, *tz), TimeFormatError);  // FIXME
  EXPECT_EQ("28",               TimeFormat("%d")(time, *tz));
  EXPECT_THROW(TimeFormat("%D")(time, *tz), TimeFormatError);  // FIXME
  EXPECT_EQ("13",               TimeFormat("%g")(time, *tz));
  EXPECT_EQ("2013",             TimeFormat("%G")(time, *tz));
  EXPECT_EQ("15",               TimeFormat("%H")(time, *tz));
  EXPECT_EQ("209",              TimeFormat("%j")(time, *tz));
  EXPECT_EQ(".000 000 000 000", TimeFormat(".%k %K %l %L")(time, *tz));
  EXPECT_EQ("07",               TimeFormat("%m")(time, *tz));
  EXPECT_EQ("37",               TimeFormat("%M")(time, *tz));
  EXPECT_EQ("UTC-14400 secs",   TimeFormat("UTC%o secs")(time, *tz));
  EXPECT_EQ("PM",               TimeFormat("%p")(time, *tz));
  EXPECT_EQ("UTC-04h, 00m",     TimeFormat("UTC%U%Qh, %qm")(time, *tz));
  EXPECT_EQ("38",               TimeFormat("%S")(time, *tz));
  EXPECT_THROW(TimeFormat("%T")(time, *tz), TimeFormatError);  // FIXME
  EXPECT_EQ("-",                TimeFormat("%U")(time, *tz));
  EXPECT_EQ("week 30 of 2013",  TimeFormat("week %V of %G")(time, *tz));
  EXPECT_EQ("0 = Sunday (Sun)", TimeFormat("%w = %W (%~W)")(time, *tz));
  EXPECT_EQ("13",               TimeFormat("%y")(time, *tz));
  EXPECT_EQ("2013",             TimeFormat("%Y")(time, *tz));
  EXPECT_THROW(TimeFormat("%Z")(time, *tz), TimeFormatError);  // FIXME
  EXPECT_EQ("EDT",              TimeFormat("%~Z")(time, *tz));

  // One Time tick is a bit less than 15 nsec.
  auto const time1 = Time::from_offset(time.get_offset() + 1);
  EXPECT_EQ(".000 000 014", TimeFormat(".%k %K %l")(time1));
}

TEST(TimeFormat, width) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JAN/1, Daytime(6, 7, 8.0), *tz);
  EXPECT_EQ("6 hr 7 min 8 sec",         TimeFormat("%0H hr %0M min %0S sec")(time, *tz));
  EXPECT_EQ("006 hr 007 min 008 sec",   TimeFormat("%3H hr %3M min %3S sec")(time, *tz));
  EXPECT_EQ("002013/001/001",           TimeFormat("%6Y/%3m/%3d")(time, *tz));
  EXPECT_EQ("0001",                     TimeFormat("%04m")(time, *tz));
  EXPECT_EQ("000000000001",             TimeFormat("%12m")(time, *tz));
}

TEST(TimeFormat, str_width) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JAN/1, Daytime(6, 7, 8.0), *tz);
  EXPECT_EQ("     TUESDAY", TimeFormat("%12^W")(time, *tz));
  EXPECT_EQ("         TUE", TimeFormat("%12^~W")(time, *tz));
  EXPECT_EQ("*****JANUARY", TimeFormat("%#*12^b")(time, *tz));
  EXPECT_EQ("*********JAN", TimeFormat("%#*12^~b")(time, *tz));
}

TEST(TimeFormat, rounding) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JAN/1, Daytime(6, 7, 8.999999), *tz);
  EXPECT_EQ("06:07:08",         TimeFormat("%H:%M:%S")(time, *tz));
  EXPECT_EQ("06:07:08.9",       TimeFormat("%H:%M:%.1S")(time, *tz));
  EXPECT_EQ("06:07:08.99",      TimeFormat("%H:%M:%.2S")(time, *tz));
  EXPECT_EQ("06:07:08.999",     TimeFormat("%H:%M:%.3S")(time, *tz));
  EXPECT_EQ("06:07:08.999999",  TimeFormat("%H:%M:%.6S")(time, *tz));
}

TEST(TimeFormat, precision) { 
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JAN/1, Daytime(6, 7, 8.01234567), *tz);
  EXPECT_EQ("06:07:08",         TimeFormat("%H:%M:%S")(time, *tz));
  EXPECT_EQ("06:07:08.",        TimeFormat("%H:%M:%.0S")(time, *tz));
  EXPECT_EQ("06:07:08.01",      TimeFormat("%H:%M:%.2S")(time, *tz));
  EXPECT_EQ("06:07:08.0123",    TimeFormat("%H:%M:%.4S")(time, *tz));
  EXPECT_EQ("06:07:08.012345",  TimeFormat("%H:%M:%.6S")(time, *tz));
  EXPECT_EQ("06:07:08.0123456", TimeFormat("%H:%M:%.7S")(time, *tz));
  EXPECT_EQ("8.0123",           TimeFormat("%1.4S")(time, *tz));
  EXPECT_EQ("08.0123",          TimeFormat("%2.4S")(time, *tz));
  EXPECT_EQ("0008.0123",        TimeFormat("%4.4S")(time, *tz));
}

TEST(TimeFormat, precision_zero) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JAN/1, Daytime(6, 7, 8), *tz);
  EXPECT_EQ("06:07:08",         TimeFormat("%H:%M:%S")(time, *tz));
  EXPECT_EQ("06:07:08.",        TimeFormat("%H:%M:%.0S")(time, *tz));
  EXPECT_EQ("06:07:08.00",      TimeFormat("%H:%M:%.2S")(time, *tz));
  EXPECT_EQ("06:07:08.0000",    TimeFormat("%H:%M:%.4S")(time, *tz));
  EXPECT_EQ("06:07:08.000000",  TimeFormat("%H:%M:%.6S")(time, *tz));
  EXPECT_EQ("06:07:08.0000000", TimeFormat("%H:%M:%.7S")(time, *tz));
  EXPECT_EQ("8.0000",           TimeFormat("%1.4S")(time, *tz));
  EXPECT_EQ("08.0000",          TimeFormat("%2.4S")(time, *tz));
  EXPECT_EQ("0008.0000",        TimeFormat("%4.4S")(time, *tz));
}

TEST(TimeFormat, pad) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JAN/1, Daytime(6, 7, 8.01234), *tz);
  EXPECT_EQ("06:07:08",         TimeFormat("%H:%M:%S")(time, *tz));
  EXPECT_EQ(" 6: 7: 8",         TimeFormat("%# H:%# M:%# S")(time, *tz));
  EXPECT_EQ("%6 $7 %8",         TimeFormat("%#%H %#$M %#%S")(time, *tz));
  EXPECT_EQ("ooooo6:ooooo7:o8.0123", TimeFormat("%6#oH:%#o6M:%#o.4S")(time, *tz));
}

TEST(TimeFormat, str_case) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JAN/1, Daytime(6, 7, 8.01234), *tz);
  EXPECT_EQ("1 = JAN",          TimeFormat("%0m = %^~b")(time));
  EXPECT_EQ("1 = JAN",          TimeFormat("%0m = %~^b")(time));
  EXPECT_EQ("1 = jan",          TimeFormat("%0m = %_~b")(time));
  EXPECT_EQ("1 = jan",          TimeFormat("%0m = %~_b")(time));
  EXPECT_EQ("1 = JANUARY",      TimeFormat("%0m = %^b")(time));

  EXPECT_EQ("2 = Tue",          TimeFormat("%0w = %~W")(time));
  EXPECT_EQ("2 = TUE",          TimeFormat("%0w = %^~W")(time));
  EXPECT_EQ("2 = TUE",          TimeFormat("%0w = %~^W")(time));
  EXPECT_EQ("2 = tue",          TimeFormat("%0w = %_~W")(time));
  EXPECT_EQ("2 = tue",          TimeFormat("%0w = %~_W")(time));
  EXPECT_EQ("2 = TUESDAY",      TimeFormat("%0w = %^W")(time));
  EXPECT_EQ("2 = Tuesday",      TimeFormat("%0w = %W")(time));
}

TEST(TimeFormat, display_time_zone) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/28, Daytime(15, 37, 38.0), *tz);

  TimeFormat const format0 = "%Y-%m-%d";
  TimeFormat const format1 = "%H:%M:%S %~Z";

  set_display_time_zone("US/Eastern");
  EXPECT_EQ("2013-07-28",       format0(time));
  EXPECT_EQ("15:37:38 EDT",     format1(time, DTZ));

  set_display_time_zone("US/Pacific");
  EXPECT_EQ("2013-07-28",       format0(time));
  EXPECT_EQ("12:37:38 PDT",     format1(time, DTZ));
}

TEST(TimeFormat, iso) {
  auto const tz = get_time_zone("US/Eastern");
  auto const time = from_local(2013/JUL/28, Daytime(15, 37, 38.0), *tz);
  set_display_time_zone("US/Eastern");
  EXPECT_EQ("20130728T153738",              TimeFormat::ISO_LOCAL_BASIC(time, DTZ));
  EXPECT_EQ("2013-07-28T15:37:38",          TimeFormat::ISO_LOCAL_EXTENDED(time, DTZ));
  EXPECT_EQ("20130728T193738Z",             TimeFormat::ISO_ZONE_LETTER_BASIC(time));
  EXPECT_EQ("2013-07-28T19:37:38Z",         TimeFormat::ISO_ZONE_LETTER_EXTENDED(time));
  EXPECT_EQ("20130728T153738-0400",         TimeFormat::ISO_ZONE_BASIC(time, DTZ));
  EXPECT_EQ("2013-07-28T15:37:38-04:00",    TimeFormat::ISO_ZONE_EXTENDED(time, DTZ));
}

TEST(TimeFormat, iso_invalid) {
  EXPECT_EQ("INVALID        ",              TimeFormat::ISO_LOCAL_BASIC(Time::INVALID));
  EXPECT_EQ("MISSING            ",          TimeFormat::ISO_LOCAL_EXTENDED(Time::MISSING));
  EXPECT_EQ("MISSING         ",             TimeFormat::ISO_ZONE_LETTER_BASIC(Time::MISSING));
  EXPECT_EQ("INVALID             ",         TimeFormat::ISO_ZONE_LETTER_EXTENDED(Time::INVALID));
  EXPECT_EQ("INVALID             ",         TimeFormat::ISO_ZONE_BASIC(Time::INVALID));
  EXPECT_EQ("MISSING                  ",    TimeFormat::ISO_ZONE_EXTENDED(Time::MISSING));
}

//------------------------------------------------------------------------------
// Class DaytimeFormat
//------------------------------------------------------------------------------

TEST(DaytimeFormat, iso) {
  Daytime const daytime(14, 5, 17.7890123456);
  EXPECT_EQ("140517",               DaytimeFormat::ISO_BASIC(daytime));
  EXPECT_EQ("14:05:17",             DaytimeFormat::ISO_EXTENDED(daytime));
  EXPECT_EQ("140517.789",           DaytimeFormat::ISO_BASIC_MSEC(daytime));
  EXPECT_EQ("14:05:17.789",         DaytimeFormat::ISO_EXTENDED_MSEC(daytime));
  EXPECT_EQ("140517.789012",        DaytimeFormat::ISO_BASIC_USEC(daytime));
  EXPECT_EQ("14:05:17.789012",      DaytimeFormat::ISO_EXTENDED_USEC(daytime));
  EXPECT_EQ("140517.789012345",     DaytimeFormat::ISO_BASIC_NSEC(daytime));
  EXPECT_EQ("14:05:17.789012345",   DaytimeFormat::ISO_EXTENDED_NSEC(daytime));
}

TEST(DaytimeFormat, iso_invalid) {
  EXPECT_EQ("INVALD",               DaytimeFormat::ISO_BASIC(Daytime::INVALID));
  EXPECT_EQ("MISSNG",               DaytimeFormat::ISO_BASIC(Daytime::MISSING));
  EXPECT_EQ("INVALID ",             DaytimeFormat::ISO_EXTENDED(Daytime::INVALID));
  EXPECT_EQ("MISSING   ",           DaytimeFormat::ISO_BASIC_MSEC(Daytime::MISSING));
  EXPECT_EQ("INVALID     ",         DaytimeFormat::ISO_EXTENDED_MSEC(Daytime::INVALID));
  EXPECT_EQ("INVALID      ",        DaytimeFormat::ISO_BASIC_USEC(Daytime::INVALID));
  EXPECT_EQ("MISSING        ",      DaytimeFormat::ISO_EXTENDED_USEC(Daytime::MISSING));
  EXPECT_EQ("INVALID         ",     DaytimeFormat::ISO_BASIC_NSEC(Daytime::INVALID));
  EXPECT_EQ("INVALID           ",   DaytimeFormat::ISO_EXTENDED_NSEC(Daytime::INVALID));
}

