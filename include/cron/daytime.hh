#pragma once

#include <cmath>
#include <limits>
#include <string>

#include "cron/types.hh"

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

extern inline bool
daytick_is_valid(
  Daytick daytick)
{
  return in_interval(DAYTICK_MIN, daytick, DAYTICK_MAX);
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

  typedef typename TRAITS::Offset Offset;

  static DaytimeTemplate constexpr MIN          = 0;
  static DaytimeTemplate constexpr MAX          = TRAITS::denominator * SECS_PER_DAY;
  static DaytimeTemplate constexpr LAST         = TRAITS::denominator * SECS_PER_DAY - 1;
  static DaytimeTemplate constexpr INVALID      = TRAITS::denominator * SECS_PER_DAY;
  static DaytimeTemplate constexpr MISSING      = TRAITS::denominator * SECS_PER_DAY + 1;
  static bool constexpr            USE_INVALID  = TRAITS::use_invalid;

  // Constructors

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
      in_interval(MIN.offset_, offset, MAX.offset_)
      ? offset
      : on_error<InvalidDaytimeError>());
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
    if (! is_valid())
      return DaytimeParts::get_invalid();

    Offset const minutes = offset_ / (SECS_PER_MIN * TRAITS::denominator);
    Offset const seconds = offset_ % (SECS_PER_MIN * TRAITS::denominator);
    return {
      (Hour) (minutes / MINS_PER_HOUR),
      (Minute) (minutes % MINS_PER_HOUR),
      (Second) seconds / TRAITS::denominator};
  }

  bool is_valid()   const { return in_interval(MIN.offset_, offset_, MAX.offset_); }
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


//------------------------------------------------------------------------------
// Static attributes
//------------------------------------------------------------------------------

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MIN;

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::LAST;

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MAX;

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::INVALID;

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MISSING;

//------------------------------------------------------------------------------
// Concrete Daytime types.
//------------------------------------------------------------------------------

// FIXME: Add static_assert that offset and denominator can represent all
// daytimes.

struct DaytimeTraits
{
  typedef uint64_t Offset;

  static Offset constexpr denominator = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
  static bool   constexpr use_invalid = true;
};

typedef DaytimeTemplate<DaytimeTraits> Daytime;


struct SafeDaytimeTraits
{
  typedef uint64_t Offset;

  static Offset constexpr denominator = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
  static bool   constexpr use_invalid = false;
};

typedef DaytimeTemplate<SafeDaytimeTraits> SafeDaytime;


struct SmallDaytimeTraits
{
  typedef uint32_t Offset;

  static Offset constexpr denominator = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
  static bool   constexpr use_invalid = true;
};

typedef DaytimeTemplate<SmallDaytimeTraits> SmallDaytime;


//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs

