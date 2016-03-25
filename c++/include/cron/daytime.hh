#pragma once

#include <cmath>
#include <limits>
#include <string>

#include "cron/types.hh"

namespace cron {

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

extern inline bool
daytick_is_valid(
  Daytick daytick)
{
  return in_range(DAYTICK_MIN, daytick, DAYTICK_MAX);
}


/**
 * Not aware of leap hours.
 */
extern inline bool
hms_is_valid(
  Hour hour,
  Minute minute,
  Second second)
{
  return 
    hour_is_valid(hour)
    && minute_is_valid(minute)
    && second_is_valid(second);
}


extern inline Daytick
hms_to_daytick(
  Hour hour,
  Minute minute,
  Second second)
{
  return (hour * SECS_PER_HOUR + minute * SECS_PER_MIN + second) * DAYTICK_PER_SEC;
}


//------------------------------------------------------------------------------

template<class TRAITS>
class DaytimeTemplate
{
public:

  using Offset = typename TRAITS::Offset;

  static Offset      constexpr DENOMINATOR = TRAITS::denominator;

  static DaytimeTemplate const MIN;
  static DaytimeTemplate const MAX;
  static DaytimeTemplate const INVALID;
  static DaytimeTemplate const MISSING;
  static bool constexpr USE_INVALID = TRAITS::use_invalid;

  static DaytimeTemplate const MIDNIGHT;

  // Constructors

  DaytimeTemplate()
    : DaytimeTemplate(INVALID)
  {
  }

  // FIXME: Get rid of this; use from_parts() instead.
  DaytimeTemplate(
    Hour hour, 
    Minute minute, 
    Second second)
    : DaytimeTemplate(hms_to_offset(hour, minute, second))
  {
  }

  DaytimeTemplate(
    DaytimeParts const& parts) 
    : DaytimeTemplate(parts.hour, parts.minute, parts.second) 
  {
  }

  static DaytimeTemplate 
  from_offset(
    Offset offset)
  {
    return DaytimeTemplate(
      in_range(MIN.offset_, offset, MAX.offset_)
      ? offset
      : on_error<InvalidDaytimeError>());
  }

  static DaytimeTemplate
  from_parts(
    Hour const hour,
    Minute const minute,
    Second const second)
  {
    return DaytimeTemplate(hms_to_offset(hour, minute, second));
  }

  static DaytimeTemplate 
  from_daytick(
    Daytick daytick)
  {
    return DaytimeTemplate(
      daytick_is_valid(daytick) 
      ? daytick_to_offset(daytick)
      : on_error<InvalidDaytimeError>());
  }

  static DaytimeTemplate 
  from_ssm(
    Ssm ssm)
  {
    return DaytimeTemplate(
      ssm_is_valid(ssm)
      ? (Offset) round(ssm * TRAITS::denominator)
      : on_error<InvalidDaytimeError>());
  }

  // Accessors.  

  Offset get_offset() const { return offset_; }

  Daytick 
  get_daytick()
    const 
  { 
    return 
      is_valid()
      ? rescale_int<Daytick, TRAITS::denominator, DAYTICK_PER_SEC>(offset_)
      : DAYTICK_INVALID;
  }

  double get_ssm() 
    const 
  { 
    return 
      is_valid()
      ? (double) offset_ / TRAITS::denominator
      : SSM_INVALID;
  }

  DaytimeParts 
  get_parts()  
    const
  {
    if (!is_valid())
      return DaytimeParts::get_invalid();

    Offset const minutes = offset_ / (SECS_PER_MIN * TRAITS::denominator);
    Offset const seconds = offset_ % (SECS_PER_MIN * TRAITS::denominator);
    return {
      (Hour) (minutes / MINS_PER_HOUR),
      (Minute) (minutes % MINS_PER_HOUR),
      (Second) seconds / TRAITS::denominator};
  }

  bool is_valid()   const { return in_range(MIN.offset_, offset_, MAX.offset_); }
  bool is_invalid() const { return is(INVALID); }
  bool is_missing() const { return is(MISSING); }

  // Comparisons

  bool is(DaytimeTemplate const& o)         const { return offset_ == o.offset_; }
  bool operator==(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ == o.offset_; }
  bool operator!=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ != o.offset_; }
  bool operator< (DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <  o.offset_; }
  bool operator<=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <= o.offset_; }
  bool operator> (DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >  o.offset_; }
  bool operator>=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >= o.offset_; }

private:

  template<class EXC>
  static Offset 
  on_error()
  {
    if (TRAITS::use_invalid)
      return INVALID.offset_;
    else
      throw EXC();
  }

  static Offset 
  daytick_to_offset(
    Daytick daytick)
  {
    return 
      daytick_is_valid(daytick)
      ? rescale_int<Daytick, DAYTICK_PER_SEC, TRAITS::denominator>(daytick)
      : on_error<InvalidDaytimeError>();
  }

  static Offset 
  hms_to_offset(
    Hour hour, 
    Minute minute, 
    Second second)
  {
    return 
      hms_is_valid(hour, minute, second)
      ? daytick_to_offset(hms_to_daytick(hour, minute, second))
      : on_error<InvalidDaytimeError>();
  }

  constexpr 
  DaytimeTemplate(
    Offset offset) 
    : offset_(offset) 
  {
  }

  Offset offset_;

};


template<class TRAITS>
inline DaytimeTemplate<TRAITS>
operator+(
  DaytimeTemplate<TRAITS> const daytime,
  double const shift)
{
  using Daytime = DaytimeTemplate<TRAITS>;

  if (daytime.is_invalid() || daytime.is_missing())
    return daytime;
  else {
    auto offset = daytime.get_offset();
    offset += round(shift * Daytime::DENOMINATOR);
    return Daytime::from_offset(offset % (SECS_PER_DAY * Daytime::DENOMINATOR));
  }
}


template<class TRAITS>
inline DaytimeTemplate<TRAITS>
operator-(
  DaytimeTemplate<TRAITS> const daytime,
  double shift)
{
  using Daytime = DaytimeTemplate<TRAITS>;

  if (shift > SECS_PER_DAY)
    shift = fmod(shift, SECS_PER_DAY);

  if (daytime.is_invalid() || daytime.is_missing())
    return daytime;
  else {
    auto shift_offset = 
      (typename Daytime::Offset) round(shift * Daytime::DENOMINATOR);
    auto offset = daytime.get_offset();
    // Avoid a negative result.
    if (offset < shift_offset)
      offset += SECS_PER_DAY * Daytime::DENOMINATOR;
    offset -= shift_offset;
    return Daytime::from_offset(offset);
  }
}


//------------------------------------------------------------------------------
// Static attributes
//------------------------------------------------------------------------------

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MIN{0};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MAX{TRAITS::denominator * SECS_PER_DAY - 1};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::INVALID{TRAITS::denominator * SECS_PER_DAY + 1};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MISSING{TRAITS::denominator * SECS_PER_DAY};

template<class TRAITS>
DaytimeTemplate<TRAITS> const
DaytimeTemplate<TRAITS>::MIDNIGHT{0, 0, 0};

//------------------------------------------------------------------------------
// Concrete Daytime types.
//------------------------------------------------------------------------------

// FIXME: Add static_assert that offset and denominator can represent all
// daytimes.

struct DaytimeTraits
{
  using Offset = uint64_t;

  static Offset constexpr denominator = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
  static bool   constexpr use_invalid = true;
};

using Daytime = DaytimeTemplate<DaytimeTraits>;


struct SafeDaytimeTraits
{
  using Offset = uint64_t;

  static Offset constexpr denominator = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
  static bool   constexpr use_invalid = false;
};

using DateDaytime = DaytimeTemplate<SafeDaytimeTraits>;


struct SmallDaytimeTraits
{
  using Offset = uint32_t;

  static Offset constexpr denominator = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
  static bool   constexpr use_invalid = true;
};

using SmallDaytime = DaytimeTemplate<SmallDaytimeTraits>;


//------------------------------------------------------------------------------

}  // namespace cron

