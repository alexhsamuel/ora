#pragma once

#include <limits>

#include "aslib/exc.hh"
#include "aslib/math.hh"

namespace cron {
namespace time {

using namespace aslib;

//------------------------------------------------------------------------------

/**
 *  This table shows some sample configurations for time representation.  The
 *  first four columns give the number of bits used for storage, 'u' for
 *  unsigned or 's' for signed, the denominator to convert the storage
 *  representation to seconds, and the base year.  The remaining columns show
 *  the total representable range in years, the range of representable years,
 *  and the approximate time resolution.
 *
 *  Note that 10000 years (the year range supported by the library) is about
 *  3.2E+11 s, which requires 39 bits represent with 1 s resolution.
 *
 *  FIXME: Maybe NsecTime should be Time.  Who measures ns before 1900?
 *  - SmallTime -> Time32
 *  - LongTime and LongTime32 to span 1-9999
 *
 *    Bits  Sgn  Denom  Base     Years  Yr. Range  Resolution    Class
 *    ----  ---  -----  ----     -----  ---------  ----------    ------------
 *      32    u  1      1970       136  1970-2106      1  s      SmallTime
 *      32    s  1      1970       136  1902-2038      1  s      Unix32Time
 *      64    s  1      1970      many  0001-9999      1  s      Unix64Time
 *      32    u  1<< 2  1990        34  1990-2024    250 ms
 *      64    u  1<<32  1970       136  1970-2106    230 ps   
 *      64    u  1<<30  1900       544  1900-2444    930 ps      NsecTime
 *      64    u  1<<28  1200      2179  1200-3379      4 ns
 *      64    u  1<<26     1      8716  0001-8717     15 ns      Time
 *     128    u  1<<89     1    ~17000  0001-9999    < 1 ys      Time128
 */

template<class TRAITS>
class TimeTemplate
{
public:

  using Traits = TRAITS;
  using Offset = typename Traits::Offset;

  static Datenum      constexpr BASE        = Traits::base;
  static Offset       constexpr DENOMINATOR = Traits::denominator;
  static TimeTemplate const     MIN;
  static TimeTemplate const     MAX;
  static TimeTemplate const     INVALID;
  static TimeTemplate const     MISSING;
  static double       constexpr RESOLUTION  = 1.0 / Traits::denominator;

  // Constructors

  TimeTemplate() 
  : offset_(Traits::invalid)
  {
  }

  template<class TTRAITS> 
  TimeTemplate(
    TimeTemplate<TTRAITS> time)
  : TimeTemplate(convert_offset(time.get_offset(), TTRAITS::denominator, TTRAITS::base))
  {
  }

  TimeTemplate(
    Datenum datenum,
    Daytick daytick,
    TimeZone const& tz,
    bool first=true)
  : offset_(datenum_daytick_to_offset(datenum, daytick, tz, first))
  {
  }

// FIXME: Remove this.
  template<class DTRAITS, class YTRAITS>
  TimeTemplate(
    date::DateTemplate<DTRAITS> const date,
    daytime::DaytimeTemplate<YTRAITS> const daytime,
    TimeZone const& tz,
    bool first=true)
  : TimeTemplate(date.get_datenum(), daytime.get_daytick(), tz, first)
  {
  }

  TimeTemplate(
    Year year,
    Month month,
    Day day,
    Hour hour,
    Minute minute,
    Second second,
    TimeZone const& tz,
    bool first=true)
  : TimeTemplate(parts_to_offset(year, month, day, hour, minute, second, tz, first))
  {
  }

  static TimeTemplate 
  from_offset(
    Offset offset)
  {
    if (in_range(Traits::min, offset, Traits::max))
      return TimeTemplate(offset);
    else
      throw TimeRangeError();
  }

  // Comparisons

  bool is_valid()   const { return in_range(MIN.offset_, offset_, MAX.offset_); }
  bool is_invalid() const { return is(INVALID); }
  bool is_missing() const { return is(MISSING); }

  bool is(TimeTemplate const& o) const { return offset_ == o.offset_; }
  bool operator==(TimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ == o.offset_; }
  bool operator!=(TimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ != o.offset_; }
  bool operator< (TimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <  o.offset_; }
  bool operator<=(TimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <= o.offset_; }
  bool operator> (TimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >  o.offset_; }
  bool operator>=(TimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >= o.offset_; }

  // Accessors

  Offset
  get_offset()
    const
  {
    ensure_valid(*this);
    return offset_;
  }

  TimeParts 
  get_parts(
    TimeZone const& tz) 
    const
  {
    if (! is_valid()) 
      return TimeParts::get_invalid();

    TimeParts parts;

    // Look up the time zone.
    parts.time_zone = tz.get_parts(*this);
    Offset const offset = offset_ + parts.time_zone.offset * Traits::denominator;

    // Establish the date and daytime parts, using division rounded toward -inf
    // and a positive remainder.
    Datenum const datenum   
      = (int64_t) (offset / Traits::denominator) / SECS_PER_DAY 
        + (offset < 0 ? -1 : 0)
        + BASE;
    Offset const day_offset 
      = (int64_t) offset % (Traits::denominator * SECS_PER_DAY) 
        + (offset < 0 ? Traits::denominator * SECS_PER_DAY : 0);

    parts.date            = datenum_to_parts(datenum);
    parts.daytime.second  = (Second) (day_offset % (SECS_PER_MIN * Traits::denominator)) / Traits::denominator;
    Offset const minutes  = day_offset / (SECS_PER_MIN * Traits::denominator);
    parts.daytime.minute  = minutes % MINS_PER_HOUR;
    parts.daytime.hour    = minutes / MINS_PER_HOUR;
    return parts;
  }

  TimeParts get_parts(std::string const& tz_name) const { return get_parts(*get_time_zone(tz_name)); }
  TimeParts get_parts() const { return get_parts(*get_display_time_zone()); }

private:

  static Offset 
  datenum_daytick_to_offset(
    Datenum datenum,
    Daytick daytick,
    TimeZone const& tz,
    bool first)
  {
    if (! datenum_is_valid(datenum)) 
      throw InvalidDateError();
    if (! daytick_is_valid(daytick))
      throw InvalidDaytimeError();

    Offset tz_offset;
    try {
      tz_offset = tz.get_parts_local(datenum, daytick, first).offset;
    }
    catch (NonexistentLocalTime) {
      // FIXME: Don't catch and rethrow...
      throw NonexistentLocalTime();
    }

    // Below, we compute this expression with overflow checking:
    //
    //     DENOMINATOR * SECS_PER_DAY * (datenum - BASE)
    //   + (Offset) rescale_int<Daytick, DAYTICK_PER_SEC, DENOMINATOR>(daytick)
    //   - DENOMINATOR * tz_offset;

    Offset const off = 
      (Offset) rescale_int<Daytick, DAYTICK_PER_SEC, DENOMINATOR>(daytick)
      - DENOMINATOR * tz_offset;
    Offset r;
    if (   mul_overflow(DENOMINATOR * SECS_PER_DAY, (Offset) datenum - BASE, r)
        || add_overflow(r, off, r))
      throw TimeRangeError();
    else
      return r;
  }

  static Offset 
  parts_to_offset(
    Year const      year,
    Month const     month,
    Day const       day,
    Hour const      hour,
    Minute const    minute,
    Second const    second,
    TimeZone const& tz,
    bool const      first=true)
  {
    if (! ymd_is_valid(year, month, day))
      throw InvalidDateError();
    if (! hms_is_valid(hour, minute, second))
      throw InvalidDaytimeError();

    Datenum const datenum = ymd_to_datenum(year, month, day);
    Daytick const daytick = hms_to_daytick(hour, minute, second);
    return datenum_daytick_to_offset(datenum, daytick, tz, first);
  }

  template<class OFFSET0>
  static Offset
  convert_offset(
    OFFSET0 offset0,
    OFFSET0 denominator0,
    Datenum base0)
  {
    auto const offset = cron::time::convert_offset(
      offset0, denominator0, base0, DENOMINATOR, BASE);
    if (in_range(Traits::min, offset, Traits::max))
      return offset;
    else
      throw InvalidTimeError();
  }

  constexpr TimeTemplate(Offset offset) : offset_(offset) {}

  Offset offset_;

};


/*
 * If `time` is invalid, throws `InvalidTimeError`.
 */
template<class TIME>
void
ensure_valid(
  TIME const time)
{
  if (!time.is_valid())
    throw InvalidTimeError();
}


//------------------------------------------------------------------------------
// Static members
//------------------------------------------------------------------------------

template<class TRAITS>
Datenum constexpr
TimeTemplate<TRAITS>::BASE;

template<class TRAITS>
typename TimeTemplate<TRAITS>::Offset constexpr
TimeTemplate<TRAITS>::DENOMINATOR;

template<class TRAITS>
TimeTemplate<TRAITS> constexpr
TimeTemplate<TRAITS>::MIN
  {TRAITS::min};

template<class TRAITS>
TimeTemplate<TRAITS> constexpr
TimeTemplate<TRAITS>::MAX
  {TRAITS::max};

template<class TRAITS>
TimeTemplate<TRAITS> constexpr
TimeTemplate<TRAITS>::INVALID
  {TRAITS::invalid};

template<class TRAITS>
TimeTemplate<TRAITS> constexpr
TimeTemplate<TRAITS>::MISSING
  {TRAITS::missing};

template<class TRAITS>
double constexpr
TimeTemplate<TRAITS>::RESOLUTION;

//------------------------------------------------------------------------------
// Concrete time classes
//------------------------------------------------------------------------------

struct TimeTraits
{
  using Offset = uint64_t;

  static Datenum constexpr base         = 0; 
  static Offset  constexpr denominator  = (Offset) 1 << 26;
  static Offset  constexpr invalid      = std::numeric_limits<Offset>::max();
  static Offset  constexpr missing      = std::numeric_limits<Offset>::max() - 1;
  static Offset  constexpr min          = 0;
  static Offset  constexpr max          = std::numeric_limits<Offset>::max() - 2;
};

extern template class TimeTemplate<TimeTraits>;
using Time =  TimeTemplate<TimeTraits>;


struct SmallTimeTraits
{
  using Offset = uint32_t;

  static Datenum constexpr base         = DATENUM_UNIX_EPOCH; 
  static Offset  constexpr denominator  = 1;
  static Offset  constexpr invalid      = std::numeric_limits<Offset>::max();
  static Offset  constexpr missing      = std::numeric_limits<Offset>::max() - 1;
  static Offset  constexpr min          = 0;
  static Offset  constexpr max          = std::numeric_limits<Offset>::max() - 2;
};

extern template class TimeTemplate<SmallTimeTraits>;
using SmallTime = TimeTemplate<SmallTimeTraits>;


struct NsecTimeTraits
{
  using Offset = uint64_t;

  static Datenum constexpr base         = 693595;  // 1900-01-01
  static Offset  constexpr denominator  = (Offset) 1 << 30;
  static Offset  constexpr invalid      = std::numeric_limits<Offset>::max();
  static Offset  constexpr missing      = std::numeric_limits<Offset>::max() - 1;
  static Offset  constexpr min          = 0;
  static Offset  constexpr max          = std::numeric_limits<Offset>::max() - 2;
};

extern template class TimeTemplate<NsecTimeTraits>;
using NsecTime = TimeTemplate<NsecTimeTraits>;


struct Unix32TimeTraits
{
  using Offset = int32_t;

  static Datenum constexpr base         = DATENUM_UNIX_EPOCH;
  static Offset  constexpr denominator  = 1;
  static Offset  constexpr invalid      = std::numeric_limits<Offset>::max();
  static Offset  constexpr missing      = std::numeric_limits<Offset>::max() - 1;
  static Offset  constexpr min          = std::numeric_limits<Offset>::min();
  static Offset  constexpr max          = std::numeric_limits<Offset>::max() - 2;
};

extern template class TimeTemplate<Unix32TimeTraits>;
using Unix32Time = TimeTemplate<Unix32TimeTraits>;


struct Unix64TimeTraits
{
  using Offset = int64_t;

  static Datenum constexpr base         = DATENUM_UNIX_EPOCH;
  static Offset  constexpr denominator  = 1;
  static Offset  constexpr min          = -62135596800l;    // 0001-01-01
  static Offset  constexpr max          = 253402300800l;    // 9999-12-31
  static Offset  constexpr invalid      = 253402300802l;
  static Offset  constexpr missing      = 253402300801l;
};

extern template class TimeTemplate<Unix64TimeTraits>;
using Unix64Time = TimeTemplate<Unix64TimeTraits>;


struct Time128Traits
{
  using Offset = uint128_t;

  static Datenum constexpr base         = 0;
  static Offset  constexpr denominator  = make_uint128(0x2000000, 0); 
                                                            // 1 << 89
  static Offset  constexpr min          = 0;                // 0001-01-01
  static Offset  constexpr max          = make_uint128(0x92ef0c7100000000, 0); 
                                                            // 9999-12-31
  static Offset  constexpr invalid      = make_uint128(0xffffffffffffffff, 0xffffffffffffffff);
  static Offset  constexpr missing      = make_uint128(0xffffffffffffffff, 0xfffffffffffffffe);
};

extern template class TimeTemplate<Time128Traits>;
using Time128 = TimeTemplate<Time128Traits>;


//------------------------------------------------------------------------------

}  // namespace time
}  // namespace cron

