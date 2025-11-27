#pragma once

#include <cmath>
#include <limits>
#include <unistd.h>

#include "ora/lib/exc.hh"
#include "ora/lib/math.hh"

namespace ora {

using namespace ora::lib;

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
Second constexpr    SECOND_END          = 60.0;
Second constexpr    SECOND_MAX          = 59.999999999999993;  // = std::nextafter(SECOND_END, SECOND_MIN)
Second constexpr    SECOND_INVALID      = std::numeric_limits<Second>::quiet_NaN();
inline constexpr bool second_is_valid(Second second) { return in_interval(SECOND_MIN, second, SECOND_END); }

using Minute = uint8_t;
Minute constexpr    MINUTE_MIN          = 0;
Minute constexpr    MINUTE_END          = 60;
Minute constexpr    MINUTE_MAX          = MINUTE_END - 1;
Minute constexpr    MINUTE_INVALID      = std::numeric_limits<Minute>::max();
inline constexpr bool minute_is_valid(Minute minute) { return in_interval(MINUTE_MIN, minute, MINUTE_END); }

using Hour = uint8_t;
Hour constexpr      HOUR_MIN            = 0;
Hour constexpr      HOUR_END            = 24;
Hour constexpr      HOUR_MAX            = HOUR_END - 1;
Hour constexpr      HOUR_INVALID        = std::numeric_limits<Hour>::max();
inline constexpr bool hour_is_valid(Hour hour) { return in_interval(HOUR_MIN, hour, HOUR_END); }

using Day = uint8_t;
Day constexpr       DAY_MIN             = 1;
Day constexpr       DAY_END             = 32;
Day constexpr       DAY_MAX             = DAY_END - 1;
Day constexpr       DAY_INVALID         = std::numeric_limits<Day>::max();
// FIXME: Maybe we shouldn't have this.
inline constexpr bool day_is_valid(Day day) { return in_interval(DAY_MIN, day, DAY_END); }

using Month = uint8_t;
Month constexpr     MONTH_MIN           = 1;
Month constexpr     MONTH_MAX           = 12;
Month constexpr     MONTH_END           = MONTH_MAX + 1;
Month constexpr     MONTH_INVALID       = std::numeric_limits<Month>::max();
inline constexpr bool month_is_valid(Month month) { return in_interval(MONTH_MIN, month, MONTH_END); }

using Year = int16_t;
Year constexpr      YEAR_MIN            = 1;
Year constexpr      YEAR_END            = 10000;
Year constexpr      YEAR_MAX            = YEAR_END - 1;
Year constexpr      YEAR_INVALID        = std::numeric_limits<Year>::min();
inline constexpr bool year_is_valid(Year year) { return in_interval(YEAR_MIN, year, YEAR_END); }

using Ordinal = uint16_t;
Ordinal constexpr   ORDINAL_MIN         = 1;
Ordinal constexpr   ORDINAL_END         = 367;
Ordinal constexpr   ORDINAL_MAX         = ORDINAL_END - 1;
Ordinal constexpr   ORDINAL_INVALID     = std::numeric_limits<Ordinal>::max();
inline constexpr bool ordinal_is_valid(Ordinal ordinal) { return in_interval(ORDINAL_MIN, ordinal, ORDINAL_END); }

using Week = uint8_t;
Week constexpr      WEEK_MIN            = 1;
Week constexpr      WEEK_END            = 54;
Week constexpr      WEEK_MAX            = WEEK_END - 1;
Week constexpr      WEEK_INVALID        = std::numeric_limits<Week>::max();
inline constexpr bool week_is_valid(Week week) { return in_interval(WEEK_MIN, week, WEEK_END); }

using Weekday = uint8_t;
Weekday constexpr   WEEKDAY_MIN         = 0;
Weekday constexpr   WEEKDAY_END         = 7;
Weekday constexpr   WEEKDAY_MAX         = WEEKDAY_END - 1;
Weekday constexpr   WEEKDAY_INVALID     = std::numeric_limits<Weekday>::max();
inline constexpr bool weekday_is_valid(Weekday weekday) { return in_interval(WEEKDAY_MIN, weekday, WEEKDAY_END); }

Weekday constexpr   MONDAY              = 0;
Weekday constexpr   TUESDAY             = 1;
Weekday constexpr   WEDNESDAY           = 2;
Weekday constexpr   THURSDAY            = 3;
Weekday constexpr   FRIDAY              = 4;
Weekday constexpr   SATURDAY            = 5;
Weekday constexpr   SUNDAY              = 6;

// Internally, daytime computations are performed on "dayticks" per midnight.  
// A daytick is defined by DAYTICKS_PER_SECOND: daytick = 1/(1<<46) s ~= 14 fs.
using Daytick = uint64_t;
Daytick constexpr   DAYTICK_PER_SEC     = (Daytick) 1 << (8 * sizeof(Daytick) - SECS_PER_DAY_BITS);
Second constexpr    DAYTICK_SEC         = 1. / DAYTICK_PER_SEC;
Daytick constexpr   DAYTICK_MIN         = 0;
Daytick constexpr   DAYTICK_END         = SECS_PER_DAY * DAYTICK_PER_SEC;
Daytick constexpr   DAYTICK_MAX         = DAYTICK_END - 1;
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
Datenum constexpr   DATENUM_END         = DATENUM_MAX + 1;
Datenum constexpr   DATENUM_INVALID     = std::numeric_limits<Datenum>::max();
Datenum constexpr   DATENUM_UNIX_EPOCH  =  719162;   // 1970-01-01
inline bool constexpr datenum_is_valid(Datenum datenum) { return in_interval(DATENUM_MIN, datenum, DATENUM_END); }

/*
 * Traditional UNIX time representation: seconds since midnight 1970-01-01 UTC.
 */
using EpochTime = int64_t;
EpochTime constexpr EPOCH_TIME_INVALID  = std::numeric_limits<EpochTime>::min();
EpochTime constexpr EPOCH_TIME_MIN      = -62135596800;  // 0001-01-01T00:00:00+00:00
EpochTime constexpr EPOCH_TIME_MAX      = 253402300799;  // 9999-12-31T23:59:59+00:00
EpochTime constexpr EPOCH_TIME_END      = EPOCH_TIME_MAX + 1;

/*
 * YMDI, a year-month-day integer.
 *
 * A YMDI encodes year, month, day as eight decimal digits YYYYMMDD.  To avoid
 * confusion with other integer representations, by convention we restrict a
 * YMDI years from 1000 to 9999.
 */
int constexpr       YMDI_MIN            = 10000000;
int constexpr       YMDI_MAX            = 99999999;
int constexpr       YMDI_END            = YMDI_MAX + 1;
int constexpr       YMDI_INVALID        = std::numeric_limits<int>::min();

/*
 * HMSF, a hours-minutes-seonds floating-point value. 
 */
double constexpr    HMSF_MIN            =      0.0;
double constexpr    HMSF_END            = 240000.0;
double constexpr    HMSF_MAX            = 235959.9999999999999;  // FIXME
double constexpr    HMSF_INVALID        = std::numeric_limits<double>::quiet_NaN();

/**
 * Seconds since midnight.
 *
 * Not aware of DST transitions.  Can represent elapsed time between 0 and 86400
 * seconds.
 */
using Ssm = double;
Ssm constexpr       SSM_MIN             = 0;
Ssm constexpr       SSM_END             = SECS_PER_DAY;
Ssm constexpr       SSM_MAX             = 86399.999999999985;  // = std::nextafter(SSM_END, SSM_MIN)
Ssm constexpr       SSM_INVALID         = std::numeric_limits<Ssm>::quiet_NaN();
inline bool constexpr ssm_is_valid(Ssm ssm) { return in_interval(SSM_MIN, ssm, SSM_END); }

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

//------------------------------------------------------------------------------

/*
 * Components of an ISO-8601 ordinal date.
 */
struct OrdinalDate
{
  Year              year                = YEAR_INVALID;
  Ordinal           ordinal             = ORDINAL_INVALID;
};


/* 
 * Components of a conventional (year, month, day) date.
 */
struct YmdDate
{
  Year              year                = YEAR_INVALID;
  Month             month               = MONTH_INVALID;
  Day               day                 = DAY_INVALID;
};


/*
 * Components of an ISO-8601 week date.
 */
struct WeekDate
{
  Year              week_year           = YEAR_INVALID;
  Week              week                = WEEK_INVALID;
  Weekday           weekday             = WEEKDAY_INVALID;
};


/*
 * Components of the various date representation.s
 */
struct FullDate
{
  OrdinalDate       ordinal_date        = {};
  YmdDate           ymd_date            = {};
  WeekDate          week_date           = {};
};


/*
 * Components of a conventional (hour, minute, second) daytime.
 */
struct HmsDaytime
{
  Hour              hour                = HOUR_INVALID;
  Minute            minute              = MINUTE_INVALID;
  Second            second              = SECOND_INVALID;
};


/*
 * As above, but no alignment padding.
 */
#pragma pack(1)
struct HmsDaytimePacked
{
  Hour              hour                = HOUR_INVALID;
  Minute            minute              = MINUTE_INVALID;
  Second            second              = SECOND_INVALID;
};
#pragma pack()

static_assert(sizeof(HmsDaytimePacked) == 10, "HmsDaytimePacked isn't 10 bytes");


inline bool constexpr 
hms_is_valid(
  HmsDaytime const& hms)
{
  return 
       hour_is_valid(hms.hour) 
    && minute_is_valid(hms.minute) 
    && second_is_valid(hms.second);
}


/*
 * The state of a time zone at a specific time.
 */
struct TimeZoneParts
{
  TimeZoneOffset    offset              = TIME_ZONE_OFFSET_INVALID;
  char              abbreviation[15]    = "?TZ";  // FIXME: ?!
  bool              is_dst              = false;
};


/*
 * A time that has been localized to a particular time zone.
 *
 * An instance does *not* uniquely represent a time.
 */
struct LocalDatenumDaytick
{
  Datenum           datenum             = DATENUM_INVALID;
  Daytick           daytick             = DAYTICK_INVALID;
  TimeZoneParts     time_zone           = {};
};


/*
 * A time that has been localized to a particular time zone.
 *
 * An instance does *not* uniquely represent a time.
 */
template<class DATE, class DAYTIME>
struct LocalTime
{
  DATE              date                = DATE::INVALID;
  DAYTIME           daytime             = DAYTIME::INVALID;
  TimeZoneParts     time_zone           = {};
};


/*
 * Components of a time that has been localized to a particular time zone.
 */
struct TimeParts
{
  YmdDate           date                = {};
  HmsDaytime        daytime             = {};
  TimeZoneParts     time_zone           = {};
};


//------------------------------------------------------------------------------

}  // namespace ora

