/*
 * Template time class.
 */

#pragma once

#include <limits>

#include "aslib/exc.hh"
#include "aslib/math.hh"
#include "cron/time_functions.hh"

namespace cron {
namespace time {

using namespace aslib;

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------

namespace safe {

template<class TIME> bool equal(TIME, TIME) noexcept;

}  // namespace safe

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
 *     128    u  1<<64     1      many  0001-9999     54 zs      Time128
 */

/*
 * Represents an approximate instant of time.
 *
 * Each template instance is a non-virtual class with a single integer data
 * member, the offset, and no nontrivial destructor behavior.  The class is
 * designed so that a pointer to the underlying integer type can safely be cast
 * to the time class.
 *
 * The offset represents the number of ticks, of a fixed resolution, since an
 * arbitrary zero point, which is UTC midnight on a given date.
 */
template<class TRAITS>
class TimeType
{
public:

  using Traits = TRAITS;
  using Offset = typename Traits::Offset;

  // Constants -----------------------------------------------------------------

  static Datenum      constexpr BASE        = Traits::base;
  static Offset       constexpr DENOMINATOR = Traits::denominator;
  static TimeType     const     MIN;
  static TimeType     const     MAX;
  static TimeType     const     INVALID;
  static TimeType     const     MISSING;
  static double       constexpr RESOLUTION  = 1.0 / Traits::denominator;

  // Constructors --------------------------------------------------------------

  // FIXME: Using '= default' causes instantiation problems?
  constexpr TimeType() noexcept {}  

  constexpr TimeType(TimeType const&) noexcept = default;

  /*
   * Constructs from another time template instance.
   *
   * If `time` is invalid or missing, constructs a corresponding invalid or
   * missing time.  If the time is valid but cannot be represented byt his time
   * type, throws <TimeRangeError>.
   */
  template<class OTHER_TRAITS> 
  TimeType(
    TimeType<OTHER_TRAITS> const time)
  : TimeType(
        time.is_invalid() ? INVALID
      : time.is_missing() ? MISSING
      : from_offset(
          // FIXME: This does not detect arithmetic overflow.
          convert_offset(
            time.get_offset(), OTHER_TRAITS::denominator, OTHER_TRAITS::base,
            DENOMINATOR, BASE)))
  {
  }

  ~TimeType() noexcept = default;

  // Factory methods  ----------------------------------------------------------

  static TimeType 
  from_offset(
    Offset const offset)
  {
    if (in_range(Traits::min, offset, Traits::max))
      return TimeType(offset);
    else
      throw TimeRangeError();
  }

  // Assignment operators ------------------------------------------------------

  TimeType
  operator=(
    TimeType const time)
    noexcept
  {
    offset_ = time.offset_;
    return *this;
  }

  template<class OTHER_TRAITS>
  TimeType
  operator=(
    TimeType<OTHER_TRAITS> const time)
  {
    return *this = TimeType(time);
  }

  // Accessors -----------------------------------------------------------------

  Offset
  get_offset()
    const
  {
    ensure_valid(*this);
    return offset_;
  }

  bool is_invalid() const noexcept { return offset_ == Traits::invalid; }
  bool is_missing() const noexcept { return offset_ == Traits::missing; }

  bool is_valid()
    const noexcept 
  { 
    return in_range(Traits::min, offset_, Traits::max); 
  }

private:

  // State ---------------------------------------------------------------------

  constexpr 
  TimeType(
    Offset const offset) 
  : offset_(offset) 
  {
  }

  Offset offset_ = Traits::invalid;

  template<class TIME> friend bool cron::time::safe::equal(TIME, TIME) noexcept;

public:

  /*
   * Returns true iff the memory layout is exactly the offset.
   */
  static bool constexpr 
  is_basic_layout()
  {
    return 
         sizeof(TimeType) == sizeof(Offset)
      && offsetof(TimeType, offset_) == 0;
  }

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
TimeType<TRAITS>::BASE;

template<class TRAITS>
typename TimeType<TRAITS>::Offset constexpr
TimeType<TRAITS>::DENOMINATOR;

template<class TRAITS>
TimeType<TRAITS> constexpr
TimeType<TRAITS>::MIN
  {TRAITS::min};

template<class TRAITS>
TimeType<TRAITS> constexpr
TimeType<TRAITS>::MAX
  {TRAITS::max};

template<class TRAITS>
TimeType<TRAITS> constexpr
TimeType<TRAITS>::INVALID
  {TRAITS::invalid};

template<class TRAITS>
TimeType<TRAITS> constexpr
TimeType<TRAITS>::MISSING
  {TRAITS::missing};

template<class TRAITS>
double constexpr
TimeType<TRAITS>::RESOLUTION;

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

extern template class TimeType<TimeTraits>;
using Time =  TimeType<TimeTraits>;
static_assert(Time::is_basic_layout(), "wrong memory layout for Time");

//------------------------------------------------------------------------------

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

extern template class TimeType<SmallTimeTraits>;
using SmallTime = TimeType<SmallTimeTraits>;
static_assert(Time::is_basic_layout(), "wrong memory layout for SmallTime");

//------------------------------------------------------------------------------

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

extern template class TimeType<NsecTimeTraits>;
using NsecTime = TimeType<NsecTimeTraits>;
static_assert(Time::is_basic_layout(), "wrong memory layout for NsecTime");

//------------------------------------------------------------------------------

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

extern template class TimeType<Unix32TimeTraits>;
using Unix32Time = TimeType<Unix32TimeTraits>;
static_assert(Time::is_basic_layout(), "wrong memory layout for Unix32Time");

//------------------------------------------------------------------------------

struct Unix64TimeTraits
{
  using Offset = int64_t;

  static Datenum constexpr base         = DATENUM_UNIX_EPOCH;
  static Offset  constexpr denominator  = 1;
  static Offset  constexpr min          = -62135596800l;    // 0001-01-01
  static Offset  constexpr max          = 253402300799l;    // 9999-12-31
  static Offset  constexpr invalid      = 253402300800l;
  static Offset  constexpr missing      = 253402300801l;
};

extern template class TimeType<Unix64TimeTraits>;
using Unix64Time = TimeType<Unix64TimeTraits>;
static_assert(Time::is_basic_layout(), "wrong memory layout for Unix64Time");

//------------------------------------------------------------------------------

struct Time128Traits
{
  using Offset = uint128_t;

  // Denominator is 2^64, so ~54 zs resolution; note that this is smaller
  // than 1 daytick.  (FIXME?)
  // 
  // Max is 1 daytick (2^-47 s ~= 7.1 fs) before 10000-01-01T00:00:00Z.

  static Datenum constexpr base         = 0;
  static Offset  constexpr denominator  = make_uint128(1, 0); 
  static Offset  constexpr min          = 0;
  static Offset  constexpr max          = make_uint128(      0x497786387f, 0xfffffffffffe0000);
  static Offset  constexpr invalid      = make_uint128(0xffffffffffffffff, 0xffffffffffffffff);
  static Offset  constexpr missing      = make_uint128(0xffffffffffffffff, 0xfffffffffffffffe);
};

extern template class TimeType<Time128Traits>;
using Time128 = TimeType<Time128Traits>;
static_assert(Time::is_basic_layout(), "wrong memory layout for Time128");

//------------------------------------------------------------------------------

}  // namespace time
}  // namespace cron

