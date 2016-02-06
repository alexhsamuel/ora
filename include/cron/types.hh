#pragma once

#include <limits>
#include <unistd.h>

#include "cron/math.hh"
#include "exc.hh"
#include "ranged.hh"

namespace alxs {
namespace cron {

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

typedef double      Second;
Second constexpr    SECOND_MIN          = 0;
Second constexpr    SECOND_MAX          = 60;
Second constexpr    SECOND_INVALID      = std::numeric_limits<Second>::quiet_NaN();
inline constexpr bool second_is_valid(Second second) { return in_interval(SECOND_MIN, second, SECOND_MAX); }

typedef uint8_t     Minute;
Minute constexpr    MINUTE_MIN          = 0;
Minute constexpr    MINUTE_MAX          = 60;
Minute constexpr    MINUTE_INVALID      = std::numeric_limits<Minute>::max();
inline constexpr bool minute_is_valid(Minute minute) { return in_interval(MINUTE_MIN, minute, MINUTE_MAX); }

typedef uint8_t     Hour;
Hour constexpr      HOUR_MIN            = 0;
Hour constexpr      HOUR_MAX            = 24;
Hour constexpr      HOUR_INVALID        = std::numeric_limits<Hour>::max();
inline constexpr bool hour_is_valid(Hour hour) { return in_interval(HOUR_MIN, hour, HOUR_MAX); }

typedef uint8_t     Day;
Day constexpr       DAY_MIN             = 0;
Day constexpr       DAY_MAX             = 31;
Day constexpr       DAY_INVALID         = std::numeric_limits<Day>::max();
inline constexpr bool day_is_valid(Day day) { return in_interval(DAY_MIN, day, DAY_MAX); }

typedef uint8_t     Month;
Month constexpr     MONTH_MIN           = 0;
Month constexpr     MONTH_MAX           = 12;
Month constexpr     MONTH_INVALID       = std::numeric_limits<Month>::max();
inline constexpr bool month_is_valid(Month month) { return in_interval(MONTH_MIN, month, MONTH_MAX); }

typedef int16_t     Year;
Year constexpr      YEAR_MIN            = 1;
Year constexpr      YEAR_MAX            = 10000;
Year constexpr      YEAR_INVALID        = std::numeric_limits<Year>::min();
inline constexpr bool year_is_valid(Year year) { return in_interval(YEAR_MIN, year, YEAR_MAX); }

typedef uint16_t    Ordinal;
Ordinal constexpr   ORDINAL_MIN         = 0;
Ordinal constexpr   ORDINAL_MAX         = 366;
Ordinal constexpr   ORDINAL_INVALID     = std::numeric_limits<Ordinal>::max();
inline constexpr bool ordinal_is_valid(Ordinal ordinal) { return in_interval(ORDINAL_MIN, ordinal, ORDINAL_MAX); }

typedef uint8_t     Week;
Week constexpr      WEEK_MIN            = 0;
Week constexpr      WEEK_MAX            = 53;
Week constexpr      WEEK_INVALID        = std::numeric_limits<Week>::max();
inline constexpr bool week_is_valid(Week week) { return in_interval(WEEK_MIN, week, WEEK_MAX); }

typedef uint8_t     Weekday;
Weekday constexpr   WEEKDAY_MIN         = 0;
Weekday constexpr   WEEKDAY_MAX         = 7;
Weekday constexpr   WEEKDAY_INVALID     = std::numeric_limits<Weekday>::max();
inline constexpr bool weekday_is_valid(Weekday weekday) { return in_interval(WEEKDAY_MIN, weekday, WEEKDAY_MAX); }

Weekday constexpr   MONDAY              = 0;
Weekday constexpr   TUESDAY             = 1;
Weekday constexpr   WEDNESDAY           = 2;
Weekday constexpr   THURSDAY            = 3;
Weekday constexpr   FRIDAY              = 4;
Weekday constexpr   SATURDAY            = 5;
Weekday constexpr   SUNDAY              = 6;


// Internally, daytime computations are performed on "dayticks" per midnight.  A
// daytick is defined by DAYTICKS_PER_SECOND.
typedef uint64_t    Daytick;
Daytick constexpr   DAYTICK_PER_SEC     = (Daytick) 1 << (8 * sizeof(Daytick) - SECS_PER_DAY_BITS);
double constexpr    DAYTICK_SEC         = 1. / DAYTICK_PER_SEC;
Daytick constexpr   DAYTICK_MIN         = 0;
Daytick constexpr   DAYTICK_MAX         = SECS_PER_DAY * DAYTICK_PER_SEC;
Daytick constexpr   DAYTICK_INVALID     = std::numeric_limits<Daytick>::max();

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
typedef uint32_t Datenum;
Datenum constexpr   DATENUM_MIN         =       0;   // 0001-01-01
Datenum constexpr   DATENUM_LAST        = 3652058;   // 9999-12-31
Datenum constexpr   DATENUM_MAX         = DATENUM_LAST + 1;
Datenum constexpr   DATENUM_INVALID     = std::numeric_limits<Datenum>::max();
Datenum constexpr   DATENUM_UNIX_EPOCH  =  719162;   // 1970-01-01
inline bool constexpr datenum_is_valid(Datenum datenum) { return in_interval(DATENUM_MIN, datenum, DATENUM_MAX); }

/**
 * Seconds since midnight.
 *
 * Not aware of DST transitions.  Can represent elapsed time between 0 and 86400
 * seconds.
 */
typedef double Ssm;
Ssm constexpr       SSM_MIN             = 0;
Ssm constexpr       SSM_MAX             = SECS_PER_DAY;
Ssm constexpr       SSM_INVALID         = std::numeric_limits<Ssm>::quiet_NaN();
inline bool constexpr ssm_is_valid(Ssm ssm) { return in_interval(SSM_MIN, ssm, SSM_MAX); }

/**
 * A time zone offset from UTC, in seconds.
 *
 * For example, U.S. Eastern Daylight Time (EDT) is four hours behind UTC, so
 * its offset is -14400.
 */
typedef int32_t TimeZoneOffset;
TimeZoneOffset constexpr TIME_ZONE_OFFSET_MIN       = -43200;
TimeZoneOffset constexpr TIME_ZONE_OFFSET_MAX       =  43200;
TimeZoneOffset constexpr TIME_ZONE_OFFSET_INVALID   = std::numeric_limits<TimeZoneOffset>::max();
inline constexpr bool time_zone_offset_is_valid(TimeZoneOffset offset) { return in_interval(TIME_ZONE_OFFSET_MIN, offset, TIME_ZONE_OFFSET_MAX); }

/**
 * A time expressed in (positive or negative) seconds since the UNIX epoch,
 * midnight on 1970 January 1.
 */
typedef int64_t TimeOffset;
TimeOffset constexpr TIME_OFFSET_MIN     = -62135596800;
TimeOffset constexpr TIME_OFFSET_INVALID = std::numeric_limits<TimeOffset>::max();


//------------------------------------------------------------------------------

struct DateParts
{
  Year      year;
  Month     month;
  Day       day;
  Ordinal   ordinal;
  Year      week_year;
  Week      week;
  Weekday   weekday;

  static DateParts const& 
  get_invalid()
  {
    static DateParts const parts{YEAR_INVALID, MONTH_INVALID, DAY_INVALID, ORDINAL_INVALID, YEAR_INVALID, WEEK_INVALID, WEEKDAY_INVALID}; 
    return parts;
  }

};


struct DaytimeParts
{
  Hour      hour;
  Minute    minute;
  Second    second;

  static DaytimeParts const&
  get_invalid()
  {
    static DaytimeParts const parts{HOUR_INVALID, MINUTE_INVALID, SECOND_INVALID};
    return parts;
  }

};


struct TimeZoneParts
{
  TimeZoneOffset offset;
  char abbreviation[7];  // FIXME: ?!
  bool is_dst;

  static TimeZoneParts const&
  get_invalid()
  {
    static TimeZoneParts const parts{TIME_ZONE_OFFSET_INVALID, "?TZ", false};
    return parts;
  }

};


struct TimeParts
{
  DateParts date;
  DaytimeParts daytime;
  TimeZoneParts time_zone;

  static TimeParts const&
  get_invalid()
  {
    static TimeParts parts;
    // FIXME: Don't reinitialize every time!
    parts.date      = DateParts::get_invalid();
    parts.daytime   = DaytimeParts::get_invalid();
    parts.time_zone = TimeZoneParts::get_invalid();
    return parts;
  }
};


//------------------------------------------------------------------------------
// Exceptions
//------------------------------------------------------------------------------

class DateError
  : public ValueError
{
public:

  DateError(std::string const& message) : ValueError(message) {}
  virtual ~DateError() throw () {}

};


class DateRangeError
  : public DateError
{
public:

  DateRangeError() : DateError("date not in range") {}
  virtual ~DateRangeError() throw () {}

};


class InvalidDateError
  : public DateError
{
public:

  InvalidDateError() : DateError("invalid date") {}
  virtual ~InvalidDateError() throw () {}

};


class DaytimeError
  : public ValueError
{
public:

  DaytimeError(std::string const& message) : ValueError(message) {}
  virtual ~DaytimeError() throw () {}

};


class TimeError
  : public ValueError
{
public:

  TimeError(std::string const& message) : ValueError(message) {}
  virtual ~TimeError() throw () {}

};


class InvalidTimeError
  : public TimeError
{
public:

  InvalidTimeError() : TimeError("invalid time") {}
  virtual ~InvalidTimeError() throw() {}

};


class InvalidDaytimeError
  : public DaytimeError
{
public:

  InvalidDaytimeError() : DaytimeError("invalid daytime") {}
  virtual ~InvalidDaytimeError() throw () {}

};


class NonexistentLocalTime
  : public ValueError
{
public:

  NonexistentLocalTime() : ValueError("local time does not exist") {}
  virtual ~NonexistentLocalTime() throw () {}

};


class TimeFormatError
  : public FormatError
{
public:

  TimeFormatError(std::string const& name) : FormatError(std::string("in time pattern: ") + name) {}
  virtual ~TimeFormatError() throw () {}

};


//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs

