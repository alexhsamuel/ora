/*
 * Template daytime (time of day) class.
 */

#pragma once

#include <cstddef>

#include "cron/daytime_math.hh"
#include "cron/exceptions.hh"
#include "cron/types.hh"

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
  static DaytimeTemplate const MIDNIGHT;
  static DaytimeTemplate const MAX;
  static DaytimeTemplate const INVALID;
  static DaytimeTemplate const MISSING;

  // Constructors  -------------------------------------------------------------

  // FIXME: Using '= default' causes instantiation problems?
  constexpr DaytimeTemplate() noexcept {}

  constexpr DaytimeTemplate(DaytimeTemplate const&) noexcept = default;

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

  ~DaytimeTemplate() = default;

  // Factory methods  ----------------------------------------------------------

  static DaytimeTemplate 
  from_daytick(
    Daytick const daytick)
  {
    if (daytick_is_valid(daytick)) {
      auto const offset = 
        rescale_int<Daytick, DAYTICK_PER_SEC, DENOMINATOR>(daytick);
      return DaytimeTemplate(offset);
    }
    else
      throw InvalidDaytimeError();
  }

  static DaytimeTemplate 
  from_offset(
    Offset const offset)
  {
    if (in_range((Offset) 0, offset, MAX_OFFSET))
      return DaytimeTemplate(offset);
    else
      throw DaytimeRangeError();
  }

  // Assignment operators  -----------------------------------------------------

  DaytimeTemplate
  operator=(
    DaytimeTemplate const daytime)
    noexcept
  {
    offset_ = daytime.offset_;
    return *this;
  }

  template<class OTHER_TRAITS>
  DaytimeTemplate
  operator=(
    DaytimeTemplate<OTHER_TRAITS> const daytime)
    noexcept
  {
    return *this =
        daytime.is_invalid() ? INVALID
      : daytime.is_missing() ? MISSING
      : from_daytick(daytime.get_daytick());
  }

  // Accessors  ----------------------------------------------------------------

  Daytick 
  get_daytick() 
    const
  { 
    ensure_valid(*this);
    return rescale_int<Daytick, DENOMINATOR, DAYTICK_PER_SEC>(offset_); 
  }

  Offset 
  get_offset() 
    const 
  {
    ensure_valid(*this);
    return offset_;
  }

  bool is_invalid() const noexcept { return offset_ == INVALID_OFFSET; }
  bool is_missing() const noexcept { return offset_ == MISSING_OFFSET; }

  bool 
  is_valid() 
    const noexcept 
  { 
    return in_range<Offset>(0, offset_, MAX_OFFSET);
  }

  // Comparisons  --------------------------------------------------------------

  bool operator==(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ == o.offset_; }
  bool operator!=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ != o.offset_; }
  bool operator< (DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <  o.offset_; }
  bool operator<=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ <= o.offset_; }
  bool operator> (DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >  o.offset_; }
  bool operator>=(DaytimeTemplate const& o) const { return is_valid() && o.is_valid() && offset_ >= o.offset_; }

private:

  static Offset constexpr MAX_OFFSET = SECS_PER_DAY * DENOMINATOR - 1;
  static Offset constexpr INVALID_OFFSET = std::numeric_limits<Offset>::max();
  static Offset constexpr MISSING_OFFSET = INVALID_OFFSET - 1;

  // State  --------------------------------------------------------------------

  constexpr 
  DaytimeTemplate(
    Offset offset) 
    : offset_(offset) 
  {
  }

  Offset offset_ = INVALID_OFFSET;

public:

  /*
   * Returns true iff the type can represent all valid daytimes.
   */
  static constexpr bool
  is_complete()
  {
    return 
        (intmax_t) (SECS_PER_DAY * DENOMINATOR)
      < (intmax_t) std::numeric_limits<Offset>::max();
  }

  /*
   * Returns true iff the memory layout is exactly the offset.
   */
  static constexpr bool
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
DaytimeTemplate<TRAITS> const
DaytimeTemplate<TRAITS>::MIDNIGHT
  {0};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MAX
  {MAX_OFFSET};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::INVALID
  {INVALID_OFFSET};

template<class TRAITS>
DaytimeTemplate<TRAITS> constexpr
DaytimeTemplate<TRAITS>::MISSING
  {MISSING_OFFSET};

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

