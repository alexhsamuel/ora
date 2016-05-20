#pragma once

#include "cron/types.hh"
#include "cron/daytime_math.hh"

namespace cron {
namespace daytime {

//------------------------------------------------------------------------------

template<class TRAITS>
class DaytimeTemplate
{
public:

  using Traits = TRAITS;
  using Offset = typename Traits::Offset;

  // Constants  ----------------------------------------------------------------

  static Offset constexpr DENOMINATOR = Traits::denominator;

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
        daytime.is_invalid() ? INVALID
      : daytime.is_missing() ? MISSING
      : from_daytick(daytime.get_daytick()))
  {
  }

  /*
   * Constructs from hour, minute, second components.
   */
  DaytimeTemplate(
    Hour const hour,
    Minute const minute,
    Second const second)
  : DaytimeTemplate(from_hms(hour, minute, second))
  {
  }

  ~DaytimeTemplate() = default;

  // Factory methods  ----------------------------------------------------------

  static DaytimeTemplate 
  from_offset(
    Offset const offset)
  {
    if (in_range((Offset) 0, offset, MAX_OFFSET))
      return DaytimeTemplate(offset);
    else
      throw DaytimeRangeError();
  }

  static DaytimeTemplate
  from_hms(
    Hour const hour,
    Minute const minute,
    Second const second)
  {
    if (hms_is_valid(hour, minute, second)) {
      auto const offset =
          (hour * SECS_PER_HOUR + minute * SECS_PER_MIN) * TRAITS::denominator
        + (Offset) (second * TRAITS::denominator);
      return DaytimeTemplate(offset);
    }
    else
      throw InvalidDaytimeError();
  }

  static DaytimeTemplate
  from_hms(
    HmsDaytime const& hms)
  {
    return from_hms(hms.hour, hms.minute, hms.second);
  }

  static DaytimeTemplate 
  from_daytick(
    Daytick const daytick)
  {
    if (daytick_is_valid(daytick)) {
      auto const offset = 
        rescale_int<Daytick, DAYTICK_PER_SEC, TRAITS::denominator>(daytick);
      return DaytimeTemplate(offset);
    }
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
    return *this =
        daytime.is_invalid() ? INVALID
      : daytime.is_missing() ? MISSING
      : from_daytick(daytime.get_daytick());
  }

  // Accessors  ----------------------------------------------------------------

  bool 
  is_valid() 
    const noexcept 
  { 
    return in_range((Offset) 0, offset_, MAX_OFFSET);
  }

  bool is_invalid() const noexcept { return is(INVALID); }
  bool is_missing() const noexcept { return is(MISSING); }

  Offset 
  get_offset() 
    const 
  {
    ensure_valid(*this);
    return offset_;
  }

  Daytick 
  get_daytick() 
    const
  { 
    ensure_valid(*this);
    return rescale_int<Daytick, Traits::denominator, DAYTICK_PER_SEC>(offset_); 
  }

  // Comparisons  --------------------------------------------------------------

  // FIXME: Move into daytime_functions.hh.
  bool is(DaytimeTemplate const& o)         const { return offset_ == o.offset_; }
  bool operator==(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ == o.offset_; }
  bool operator!=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ != o.offset_; }
  bool operator< (DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <  o.offset_; }
  bool operator<=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <= o.offset_; }
  bool operator> (DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >  o.offset_; }
  bool operator>=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >= o.offset_; }

private:

  static Offset constexpr MAX_OFFSET = SECS_PER_DAY * Traits::denominator - 1;
  static Offset constexpr INVALID_OFFSET = std::numeric_limits<Offset>::max();
  static Offset constexpr MISSING_OFFSET = INVALID_OFFSET - 1;

  // State  --------------------------------------------------------------------

  constexpr 
  DaytimeTemplate(
    Offset offset) 
    : offset_(offset) 
  {
  }

  Offset offset_;

public:

  /*
   * Returns true iff the type can represent all valid daytimes.
   */
  static bool constexpr
  is_complete()
  {
    return 
         std::numeric_limits<Offset>::min() <= 0
      && SECS_PER_DAY * DENOMINATOR < (long) std::numeric_limits<Offset>::max();
  }

  /*
   * Returns true iff the memory layout is exactly the offset.
   */
  static bool constexpr
  is_basic_layout()
  {
    return
         sizeof(DaytimeTemplate) == sizeof(Offset)
      && offsetof(DaytimeTemplate, offset_) == 0;
  }

};


/*
 * If `daytime` is invalid, throws `InvalidDaytimeError`.
 */
template<class DAYTIME>
void
ensure_valid(
  DAYTIME const daytime)
{
  if (!daytime.is_valid())
    throw InvalidDaytimeError();
}


//------------------------------------------------------------------------------
// Static attributes
//------------------------------------------------------------------------------

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MIN
  {0};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MAX
  {TRAITS::denominator * SECS_PER_DAY - 1};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::INVALID
  {INVALID_OFFSET};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MISSING
  {MISSING_OFFSET};

template<class TRAITS>
DaytimeTemplate<TRAITS> const
DaytimeTemplate<TRAITS>::MIDNIGHT
  {0};

//------------------------------------------------------------------------------
// Daytime template instances
//------------------------------------------------------------------------------

struct DaytimeTraits
{
  using Offset = uint64_t;

  static Offset constexpr 
  denominator 
    = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
};

extern template class DaytimeTemplate<DaytimeTraits>;
using Daytime = DaytimeTemplate<DaytimeTraits>;
static_assert(Daytime::is_complete(), "Daytime is not complete");
static_assert(Daytime::is_basic_layout(), "wrong memory layout for Daytime");

//------------------------------------------------------------------------------

struct Daytime32Traits
{
  using Offset = uint32_t;

  static Offset constexpr 
  denominator 
    = (Offset) 1 << (8 * sizeof(Offset) - SECS_PER_DAY_BITS);
};

extern template class DaytimeTemplate<Daytime32Traits>;
using Daytime32 = DaytimeTemplate<Daytime32Traits>;
static_assert(Daytime32::is_complete(), "Daytime32 is not complete");
static_assert(Daytime32::is_basic_layout(), "wrong memory layout for Daytime32");

//------------------------------------------------------------------------------

}  // namespace daytime
}  // namespace cron

