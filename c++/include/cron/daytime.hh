#pragma once

#include <cmath>
#include <limits>
#include <string>

#include "cron/daytime_functions.hh"
#include "cron/types.hh"

namespace cron {

//------------------------------------------------------------------------------

template<class TRAITS>
class DaytimeTemplate
{
public:

  using Offset = typename TRAITS::Offset;

  // Constants  ----------------------------------------------------------------

  static Offset constexpr DENOMINATOR = TRAITS::denominator;

  static DaytimeTemplate const MIN;
  static DaytimeTemplate const MAX;
  static DaytimeTemplate const INVALID;
  static DaytimeTemplate const MISSING;

  static DaytimeTemplate const MIDNIGHT;

  // Constructors  -------------------------------------------------------------

  /*
   * Default constructor: an invalid daytime.
   */
  DaytimeTemplate()
    : DaytimeTemplate(INVALID_OFFSET)
  {
  }

  /*
   * Copy constructor.
   */
  DaytimeTemplate(
    DaytimeTemplate const& daytime)
  : offset_(daytime.offset_)
  {
  }

  /*
   * Constructs from another daytime template instance.
   */
  template<class OTHER_TRAITS>
  DaytimeTemplate(
    DaytimeTemplate<OTHER_TRAITS> const daytime)
  : DaytimeTemplate(
        daytime.is_invalid() ? TRAITS::invalid
      : daytime.is_missing() ? TRAITS::missing
      : daytick_to_offset(daytime.get_daytick()))
  {
  }

  /*
   * Constructs from hour, minute, second components.
   */
  DaytimeTemplate(
    Hour const hour,
    Minute const minute,
    Second const second)
  : offset_(hms_to_offset(hour, minute, second))
  {
  }

  // Assignment operators  -----------------------------------------------------

  DaytimeTemplate
  operator=(
    DaytimeTemplate const daytime)
  {
    offset_ = daytime.offset_;
    return *this;
  }

  template<class OTHER_TRAITS>
  DaytimeTemplate
  operator=(
    DaytimeTemplate<OTHER_TRAITS> const daytime)
  {
    offset_ = 
        daytime.is_invalid() ? TRAITS::invalid
      : daytime.is_missing() ? TRAITS::missing
      : daytick_to_offset(daytime.get_daytick());
    return *this;
  }

  // Factory methods  ----------------------------------------------------------

  static DaytimeTemplate 
  from_offset(
    Offset offset)
  {
    return DaytimeTemplate(valid_offset<>(offset));
  }

  static DaytimeTemplate
  from_hms(
    Hour const hour,
    Minute const minute,
    Second const second)
  {
    return DaytimeTemplate(hour, minute, second);
  }

  static DaytimeTemplate 
  from_daytick(
    Daytick daytick)
  {
    if (daytick_is_valid(daytick))
      return daytick_to_offset(daytick);
    else
      throw InvalidDaytimeError();
  }

  static DaytimeTemplate 
  from_ssm(
    Ssm ssm)
  {
    if (ssm_is_valid(ssm))
      return DaytimeTemplate((Offset) round(ssm * TRAITS::denominator));
    else
      throw InvalidDaytimeError();
  }

  // Accessors  ----------------------------------------------------------------

  bool is_valid()   const { return offset_is_valid(offset_); }
  bool is_invalid() const { return is(INVALID); }
  bool is_missing() const { return is(MISSING); }

  Offset get_offset() const 
    { return valid_offset(); }
  Daytick get_daytick() const
    { return offset_to_daytick(valid_offset()); }
  double get_ssm() const
    { return (double) valid_offset() / TRAITS::denominator; }
  
  HmsDaytime 
  get_hms()  
    const
  {
    auto const offset = valid_offset();
    auto const minutes = offset / (SECS_PER_MIN * TRAITS::denominator);
    auto const seconds = offset % (SECS_PER_MIN * TRAITS::denominator);
    return {
      (Hour)   (minutes / MINS_PER_HOUR),
      (Minute) (minutes % MINS_PER_HOUR),
      (Second) seconds / TRAITS::denominator
    };
  }

  // Comparisons  --------------------------------------------------------------

  bool is(DaytimeTemplate const& o)         const { return offset_ == o.offset_; }
  bool operator==(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ == o.offset_; }
  bool operator!=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ != o.offset_; }
  bool operator< (DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <  o.offset_; }
  bool operator<=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <= o.offset_; }
  bool operator> (DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >  o.offset_; }
  bool operator>=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >= o.offset_; }

private:

  static Offset constexpr INVALID_OFFSET = std::numeric_limits<Offset>::max();
  static Offset constexpr MISSING_OFFSET = INVALID_OFFSET - 1;

  // Helper methods  -----------------------------------------------------------

  static bool
  offset_is_valid(
    Offset const offset)
  {
    return in_interval((Offset) 0, offset, SECS_PER_DAY * TRAITS::denominator);
  }

  template<class EXCEPTION=InvalidDaytimeError>
  static Offset
  valid_offset(
    Offset const offset)
  {
    if (offset_is_valid(offset))
      return offset;
    else
      throw EXCEPTION();
  }

  Offset valid_offset() const 
    { return valid_offset(offset_); }

  static Offset 
  daytick_to_offset(
    Daytick const daytick)
  {
    return rescale_int<Daytick, DAYTICK_PER_SEC, TRAITS::denominator>(daytick);
  }

  static Daytick 
  offset_to_daytick(
    Offset const offset)
  {
    return rescale_int<Daytick, TRAITS::denominator, DAYTICK_PER_SEC>(offset);
  }

  static Daytick
  hms_to_offset(
    Hour const hour,
    Minute const minute,
    Second const second)
  {
    if (hms_is_valid(hour, minute, second))
      return 
          (hour * SECS_PER_HOUR + minute * SECS_PER_MIN) * TRAITS::denominator
        + (Offset) (second * TRAITS::denominator);
    else
      throw InvalidDaytimeError();
  }

  // State  --------------------------------------------------------------------

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
DaytimeTemplate<TRAITS>::INVALID{INVALID_OFFSET};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MISSING{MISSING_OFFSET};

template<class TRAITS>
DaytimeTemplate<TRAITS> const
DaytimeTemplate<TRAITS>::MIDNIGHT{0};

//------------------------------------------------------------------------------
// Concrete Daytime types.
//------------------------------------------------------------------------------

// FIXME: Add static_assert that offset and denominator can represent all
// daytimes.

struct DaytimeTraits
{
  using Offset = uint64_t;

  static Offset constexpr denominator = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
};

using Daytime = DaytimeTemplate<DaytimeTraits>;


struct Daytime32Traits
{
  using Offset = uint32_t;

  static Offset constexpr denominator = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
};

using Daytime32 = DaytimeTemplate<Daytime32Traits>;


//------------------------------------------------------------------------------

}  // namespace cron


