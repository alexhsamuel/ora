#pragma once

#include <cmath>
#include <limits>
#include <unistd.h>

#include "aslib/exc.hh"
#include "aslib/math.hh"
#include "aslib/ranged.hh"

namespace cron {

using namespace aslib;

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

uint32_t constexpr  SECS_PER_MIN        = 60;
uint32_t constexpr  MINS_PER_HOUR       = 60;
uint32_t constexpr  HOURS_PER_DAY       = 24;

uint32_t constexpr  SECS_PER_HOUR       = SECS_PER_MIN * MINS_PER_HOUR;
uint32_t constexpr  SECS_PER_DAY        = SECS_PER_HOUR * HOURS_PER_DAY;

// 17 bits are required to represent SECS_PER_DAY as an integer.  Any remaining
// available bits can be used for fractional seconds.
size_t constexpr    SECS_PER_DAY_BITS   = 17;

//------------------------------------------------------------------------------
// Types
//------------------------------------------------------------------------------

using Second = double;
Second constexpr    SECOND_MIN          = 0.0;
Second constexpr    SECOND_BOUND        = 60.0;
Second constexpr    SECOND_MAX          = 0x1.dffffffffffffp+5;  // = std::nextafter(SECOND_BOUND, SECOND_MIN)
Second constexpr    SECOND_INVALID      = std::numeric_limits<Second>::quiet_NaN();
inline constexpr bool second_is_valid(Second second) { return in_interval(SECOND_MIN, second, SECOND_BOUND); }

using Minute = uint8_t;
Minute constexpr    MINUTE_MIN          = 0;
Minute constexpr    MINUTE_BOUND        = 60;
Minute constexpr    MINUTE_MAX          = MINUTE_BOUND - 1;
Minute constexpr    MINUTE_INVALID      = std::numeric_limits<Minute>::max();
inline constexpr bool minute_is_valid(Minute minute) { return in_interval(MINUTE_MIN, minute, MINUTE_BOUND); }

using Hour = uint8_t;
Hour constexpr      HOUR_MIN            = 0;
Hour constexpr      HOUR_BOUND          = 24;
Hour constexpr      HOUR_MAX            = HOUR_BOUND - 1;
Hour constexpr      HOUR_INVALID        = std::numeric_limits<Hour>::max();
inline constexpr bool hour_is_valid(Hour hour) { return in_interval(HOUR_MIN, hour, HOUR_BOUND); }

using Day = uint8_t;
Day constexpr       DAY_MIN             = 0;
Day constexpr       DAY_BOUND           = 31;
Day constexpr       DAY_MAX             = DAY_BOUND - 1;
Day constexpr       DAY_INVALID         = std::numeric_limits<Day>::max();
// FIXME: Maybe we shouldn't have this.
inline constexpr bool day_is_valid(Day day) { return in_interval(DAY_MIN, day, DAY_BOUND); }

using Month = uint8_t;
Month constexpr     MONTH_MIN           = 0;
Month constexpr     MONTH_BOUND         = 12;
Month constexpr     MONTH_MAX           = MONTH_BOUND - 1;
Month constexpr     MONTH_INVALID       = std::numeric_limits<Month>::max();
inline constexpr bool month_is_valid(Month month) { return in_interval(MONTH_MIN, month, MONTH_BOUND); }

using Year = int16_t;
Year constexpr      YEAR_MIN            = 1;
Year constexpr      YEAR_BOUND          = 10000;
Year constexpr      YEAR_MAX            = YEAR_BOUND - 1;
Year constexpr      YEAR_INVALID        = std::numeric_limits<Year>::min();
inline constexpr bool year_is_valid(Year year) { return in_interval(YEAR_MIN, year, YEAR_BOUND); }

using Ordinal = uint16_t;
Ordinal constexpr   ORDINAL_MIN         = 0;
Ordinal constexpr   ORDINAL_BOUND       = 366;
Ordinal constexpr   ORDINAL_MAX         = ORDINAL_BOUND - 1;
Ordinal constexpr   ORDINAL_INVALID     = std::numeric_limits<Ordinal>::max();
inline constexpr bool ordinal_is_valid(Ordinal ordinal) { return in_interval(ORDINAL_MIN, ordinal, ORDINAL_BOUND); }

using Week = uint8_t;
Week constexpr      WEEK_MIN            = 0;
Week constexpr      WEEK_BOUND          = 53;
Week constexpr      WEEK_MAX            = WEEK_BOUND - 1;
Week constexpr      WEEK_INVALID        = std::numeric_limits<Week>::max();
inline constexpr bool week_is_valid(Week week) { return in_interval(WEEK_MIN, week, WEEK_BOUND); }

using Weekday = uint8_t;
Weekday constexpr   WEEKDAY_MIN         = 0;
Weekday constexpr   WEEKDAY_BOUND       = 7;
Weekday constexpr   WEEKDAY_MAX         = WEEKDAY_BOUND - 1;
Weekday constexpr   WEEKDAY_INVALID     = std::numeric_limits<Weekday>::max();
inline constexpr bool weekday_is_valid(Weekday weekday) { return in_interval(WEEKDAY_MIN, weekday, WEEKDAY_BOUND); }

Weekday constexpr   MONDAY              = 0;
Weekday constexpr   TUESDAY             = 1;
Weekday constexpr   WEDNESDAY           = 2;
Weekday constexpr   THURSDAY            = 3;
Weekday constexpr   FRIDAY              = 4;
Weekday constexpr   SATURDAY            = 5;
Weekday constexpr   SUNDAY              = 6;

// Internally, daytime computations are performed on "dayticks" per midnight.  A
// daytick is defined by DAYTICKS_PER_SECOND.
using Daytick = uint64_t;
Daytick constexpr   DAYTICK_PER_SEC     = (Daytick) 1 << (8 * sizeof(Daytick) - SECS_PER_DAY_BITS);
double constexpr    DAYTICK_SEC         = 1. / DAYTICK_PER_SEC;
Daytick constexpr   DAYTICK_MIN         = 0;
Daytick constexpr   DAYTICK_BOUND       = SECS_PER_DAY * DAYTICK_PER_SEC;
Daytick constexpr   DAYTICK_MAX         = DAYTICK_BOUND - 1;
Daytick constexpr   DAYTICK_INVALID     = std::numeric_limits<Daytick>::max();
inline constexpr bool daytick_is_valid(Daytick const daytick)
  { return in_range(DAYTICK_MIN, daytick, DAYTICK_MAX); }

/**
 * Internal representation of dates.
 *
 * We perform date computations on "datenums", the number of days elapsed since
 * 0001 January 1.  (This is before the Gregorian calendar was adopted, but we
 * use the proleptic Gregorian calendar.)  
 *
 * The minimum year is the year C.E. 1; B.C.E. years are not supported.  The
 * maximum year is 9999.  Too much code exists that assumes four-digit years to
 * make larger years worth the trouble.  Also, if we are still using the same
 * f***ed up calendar system in 8,000 years, I will kill myself.
 */
using Datenum = uint32_t;
Datenum constexpr   DATENUM_MIN         =       0;   // 0001-01-01
Datenum constexpr   DATENUM_MAX         = 3652058;   // 9999-12-31
Datenum constexpr   DATENUM_BOUND       = DATENUM_MAX + 1;
Datenum constexpr   DATENUM_INVALID     = std::numeric_limits<Datenum>::max();
Datenum constexpr   DATENUM_UNIX_EPOCH  =  719162;   // 1970-01-01
inline bool constexpr datenum_is_valid(Datenum datenum) { return in_interval(DATENUM_MIN, datenum, DATENUM_BOUND); }

/*
 * YMDI, a year-month-day integer.
 *
 * A YMDI encodes year, month, day as eight decimal digits YYYYMMDD.  To avoid
 * confusion with other integer representations, by convention we restrict a
 * YMDI years from 1000 to 9999.
 */
int constexpr       YMDI_MIN            = 10000000;
int constexpr       YMDI_MAX            = 99999999;
int constexpr       YMDI_BOUND          = YMDI_MAX + 1;
int constexpr       YMDI_INVALID        = std::numeric_limits<int>::min();

/**
 * Seconds since midnight.
 *
 * Not aware of DST transitions.  Can represent elapsed time between 0 and 86400
 * seconds.
 */
using Ssm = double;
Ssm constexpr       SSM_MIN             = 0;
Ssm constexpr       SSM_BOUND           = SECS_PER_DAY;
Ssm constexpr       SSM_MAX             = 0x1.517ffffffffffp+16;  // = std::nextafter(SSM_BOUND, SSM_MIN)
Ssm constexpr       SSM_INVALID         = std::numeric_limits<Ssm>::quiet_NaN();
inline bool constexpr ssm_is_valid(Ssm ssm) { return in_interval(SSM_MIN, ssm, SSM_BOUND); }

inline Daytick constexpr ssm_to_daytick(Ssm ssm) { return (Daytick) (DAYTICK_PER_SEC * ssm + 0.5); }

/**
 * A time zone offset from UTC, in seconds.
 *
 * For example, U.S. Eastern Daylight Time (EDT) is four hours behind UTC, so
 * its offset is -14400.
 */
using TimeZoneOffset = int32_t;
TimeZoneOffset constexpr TIME_ZONE_OFFSET_MIN       = -43200;
TimeZoneOffset constexpr TIME_ZONE_OFFSET_MAX       =  43200;
TimeZoneOffset constexpr TIME_ZONE_OFFSET_INVALID   = std::numeric_limits<TimeZoneOffset>::max();
inline constexpr bool time_zone_offset_is_valid(TimeZoneOffset offset) { return in_range(TIME_ZONE_OFFSET_MIN, offset, TIME_ZONE_OFFSET_MAX); }

// FIXME: Rename this.
/**
 * A time expressed in (positive or negative) seconds since the UNIX epoch,
 * midnight on 1970 January 1.
 */
using TimeOffset = int64_t;
TimeOffset constexpr TIME_OFFSET_MIN        = -62135596800;
TimeOffset constexpr TIME_OFFSET_INVALID    = std::numeric_limits<TimeOffset>::max();

/**
 * A time expressed in units of 1/(1 << 80) seconds since 0001-01-01T00:00:00Z.
 * Each timetick unit is slightly less than 1 yoctosecond.
 */
using Timetick = int128_t;
Timetick constexpr TIMETICK_PER_SEC         = ((int128_t) 1) << 80;
double   constexpr TIMETICK_SEC             = 1. / TIMETICK_PER_SEC;
Timetick constexpr TIMETICK_MIN             = 0;
Timetick constexpr TIMETICK_MAX             = 3652059 * SECS_PER_DAY * TIMETICK_PER_SEC;
Timetick const     TIMETICK_INVALID         = ((int128_t) -1) << 127;

//------------------------------------------------------------------------------

/*
 * Components of an ISO-8601 ordinal date.
 */
struct OrdinalDate
{
  Year      year;
  Ordinal   ordinal;

  static constexpr OrdinalDate get_invalid()
    { return {YEAR_INVALID, ORDINAL_INVALID}; }

};


/* 
 * Components of a conventional (year, month, day) date.
 */
struct YmdDate
{
  Year      year;
  Month     month;
  Day       day;

  static constexpr YmdDate get_invalid()
    { return {YEAR_INVALID, MONTH_INVALID, DAY_INVALID}; }

};


/*
 * Components of an ISO-8601 week date.
 */
struct WeekDate
{
  Year      week_year;
  Week      week;
  Weekday   weekday;

  static constexpr WeekDate get_invalid()
    { return {YEAR_INVALID, WEEK_INVALID, WEEKDAY_INVALID}; }

};


// FIXME: Get rid of this by refactoring format.cc.

struct DateParts
{
  Year      year;
  Month     month;
  Day       day;
  Ordinal   ordinal;
  Year      week_year;
  Week      week;
  Weekday   weekday;

  static constexpr DateParts 
  get_invalid()
  { 
    return {
      YEAR_INVALID, MONTH_INVALID, DAY_INVALID, 
      ORDINAL_INVALID, YEAR_INVALID, WEEK_INVALID, WEEKDAY_INVALID}; 
  }

};


struct HmsDaytime
{
  Hour      hour;
  Minute    minute;
  Second    second;

  static constexpr HmsDaytime get_invalid()
    { return {HOUR_INVALID, MINUTE_INVALID, SECOND_INVALID}; }

};


struct TimeZoneParts
{
  TimeZoneOffset offset;
  char abbreviation[7];  // FIXME: ?!
  bool is_dst;

  static constexpr TimeZoneParts get_invalid()
    { return TimeZoneParts{TIME_ZONE_OFFSET_INVALID, "?TZ", false}; }

};


struct TimeParts
{
  DateParts date;
  HmsDaytime daytime;
  TimeZoneParts time_zone;

  static constexpr TimeParts 
  get_invalid()
  { 
    return {
      DateParts::get_invalid(), 
      HmsDaytime::get_invalid(), 
      TimeZoneParts::get_invalid()}; 
  }

};


//------------------------------------------------------------------------------
// Local time structs
//------------------------------------------------------------------------------

struct LocalDatenumDaytick
{
  Datenum   datenum = DATENUM_INVALID;
  Daytick   daytick = DAYTICK_INVALID;
};


template<class DATE, class DAYTIME>
struct LocalTime
{
  DATE      date    = DATE::INVALID;
  DAYTIME   daytime = DAYTIME::INVALID;
};


//------------------------------------------------------------------------------

}  // namespace cron

