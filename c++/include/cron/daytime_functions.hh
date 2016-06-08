#pragma once

#include "aslib/math.hh"

#include "cron/types.hh"
#include "cron/daytime_type.hh"

namespace cron {
namespace daytime {

//------------------------------------------------------------------------------
// Factory functions
//------------------------------------------------------------------------------

// Synonyms for static factory methods; included for completeness.

template<class DAYTIME=Daytime> inline DAYTIME from_offset(typename DAYTIME::Offset const o)
  { return DAYTIME::from_offset(o); }
template<class DAYTIME=Daytime> inline DAYTIME from_hms(Hour const h, Minute const m, Second const s=0)
  { return DAYTIME::from_hms(h, m, s); }
template<class DAYTIME=Daytime> inline DAYTIME from_hms(HmsDaytime const& d)
  { return DAYTIME::from_hms(d); }
template<class DAYTIME=Daytime> inline DAYTIME from_daytick(Daytick const d)
  { return DAYTIME::from_daytick(d); }
template<class DAYTIME=Daytime> inline DAYTIME from_ssm(Ssm const s)
  { return DAYTIME::from_ssm(s); }

//------------------------------------------------------------------------------
// Accessors
//------------------------------------------------------------------------------

template<class DAYTIME>
inline double 
get_ssm(
  DAYTIME const daytime)
{
  ensure_valid(daytime);
  return (double) daytime.get_offset() / DAYTIME::Traits::denominator; 
}


template<class DAYTIME>
inline HmsDaytime 
get_hms(
  DAYTIME const daytime)  
{
  ensure_valid(daytime);
  auto const offset = daytime.get_offset();
  auto const minutes = offset / (SECS_PER_MIN * DAYTIME::Traits::denominator);
  auto const seconds = offset % (SECS_PER_MIN * DAYTIME::Traits::denominator);
  return {
    (Hour)   (minutes / MINS_PER_HOUR),
    (Minute) (minutes % MINS_PER_HOUR),
    (Second) seconds / DAYTIME::Traits::denominator
  };
}


// For convenience.
template<class DAYTIME> inline Hour get_hour(DAYTIME const daytime)
  { return get_hms(daytime).hour; }
template<class DAYTIME> inline Minute get_minute(DAYTIME const daytime)
  { return get_hms(daytime).minute; }
template<class DAYTIME> inline Second get_second(DAYTIME const daytime)
  { return get_hms(daytime).second; }

//------------------------------------------------------------------------------
// Arithemtic with seconds
//------------------------------------------------------------------------------

template<class DAYTIME>
inline DAYTIME
seconds_after(
  DAYTIME const daytime,
  double const seconds)
{
  ensure_valid(daytime);
  return from_offset<DAYTIME>(
    daytime.get_offset() + round(seconds * DAYTIME::DENOMINATOR));
}


template<class DAYTIME>
inline DAYTIME
seconds_before(
  DAYTIME const daytime,
  double const seconds)
{
  ensure_valid(daytime);
  return from_offset<DAYTIME>(
    daytime.get_offset() - round(seconds * DAYTIME::DENOMINATOR));
}


template<class DAYTIME>
inline double
seconds_between(
  DAYTIME const daytime0,
  DAYTIME const daytime1)
{
  ensure_valid(daytime0);
  ensure_valid(daytime1);
  return 
    ((double) daytime0.get_offset() - (double) daytime1.get_offset()) 
    / DAYTIME::DENOMINATOR;
}


template<class DAYTIME> inline DAYTIME operator+(DAYTIME const d, double const secs)
  { return seconds_after(d, secs); }
template<class DAYTIME> inline DAYTIME operator-(DAYTIME const d, double const secs)
  { return seconds_before(d, secs); }
template<class DAYTIME> inline int operator-(DAYTIME const d0, DAYTIME const d1)
  { return seconds_between(d0, d1); } 

template<class DAYTIME> inline DAYTIME operator+=(DAYTIME& d, int const secs) 
  { return d = d + secs; }
template<class DAYTIME> inline DAYTIME operator++(DAYTIME& d) 
  { return d = d + 1; }
template<class DAYTIME> inline DAYTIME operator++(DAYTIME& d, int /* tag */) 
  { auto old = d; d = d + 1; return old; }
template<class DAYTIME> inline DAYTIME operator-=(DAYTIME& d, int const secs) 
  { return d = d -secs; }
template<class DAYTIME> inline DAYTIME operator--(DAYTIME& d) 
  { return d = d - 1; }
template<class DAYTIME> inline DAYTIME operator--(DAYTIME& d, int /* tag */) 
  { auto old = d; d = d - 1; return old; }

//------------------------------------------------------------------------------

}  // namespace daytime
}  // namespace cron

