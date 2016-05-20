#pragma once

#include "cron/date.hh"
#include "cron/date_math.hh"
#include "cron/types.hh"

namespace cron {
namespace date {

//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------

template<class DATE> DATE from_ymd(YmdDate const&);

//------------------------------------------------------------------------------
// Factory functions
//------------------------------------------------------------------------------

// Synonyms for static factory methods; included for completeness.

template<class DATE> inline DATE from_datenum(Datenum const d)
  { return DATE::from_datenum(d); }
template<class DATE> inline DATE from_iso_date(std::string const& d)
  { return DATE::from_iso_date(d); }
template<class DATE> inline DATE from_offset(typename DATE::Offset const o)
  { return DATE::from_offset(o); }
template<class DATE> inline DATE from_ordinal_date(Year const y, Ordinal const o) 
  { return DATE::from_ordinal_date(y, o); }
template<class DATE> inline DATE from_week_date(Year const y, Week const w, Weekday const d)
  { return DATE::from_week_date(y, w, d); }
template<class DATE> inline DATE from_ymd(Year const y, Month const m, Day const d)
  { return DATE::from_ymd(y, m, d); }
template<class DATE> inline DATE from_ymd(YmdDate const& d)
  { return DATE::from_ymd(d); }
template<class DATE> inline DATE from_ymdi(int y)
  { return DATE::from_ymdi(y); }

//------------------------------------------------------------------------------
// Accessors
//------------------------------------------------------------------------------

template<class DATE>
inline OrdinalDate 
get_ordinal_date(
  DATE const date)
{ 
  ensure_valid(date);
  return datenum_to_ordinal_date(date.get_datenum());
}


template<class DATE>
inline Weekday 
get_weekday(
  DATE const date)
{ 
  ensure_valid(date);
  return cron::get_weekday(date.get_datenum());
}


template<class DATE>
inline WeekDate 
get_week_date(
  DATE const date)
{ 
  ensure_valid(date);
  return cron::datenum_to_week_date(date.get_datenum());
}


template<class DATE>
inline YmdDate 
get_ymd(
  DATE const date)
{ 
  ensure_valid(date);
  return datenum_to_ymd(date.get_datenum()); 
}


template<class DATE>
inline int 
get_ymdi(
  DATE const date)
{ 
  ensure_valid(date);
  return cron::datenum_to_ymdi(date.get_datenum()); 
}


// For convenience.
template<class DATE> inline Day get_day(DATE const date) 
  { return get_ymd(date).day; }
template<class DATE> inline Month get_month(DATE const date) 
  { return get_ymd(date).month; }
template<class DATE> inline Ordinal get_ordinal(DATE const date)
  { return get_ordinal_date(date).ordinal; }
template<class DATE> inline Week get_week(DATE const date)
  { return get_week_date(date).week; }
template<class DATE> inline Year get_week_year(DATE const date)
  { return get_week_date(date).week_year; }
template<class DATE> inline Year get_year(DATE const date) 
  { return get_ordinal_date(date).year; }

//------------------------------------------------------------------------------
// Day arithmetic
//------------------------------------------------------------------------------

template<class DATE>
inline DATE
days_after(
  DATE const date,
  int const days)
{
  ensure_valid(date);
  return from_offset<DATE>(date.get_offset() + days);
}


template<class DATE>
inline DATE
days_before(
  DATE const date,
  int const days)
{
  return days_after(date, -days);
}


template<class DATE>
inline int
days_between(
  DATE const date0,
  DATE const date1)
{
  ensure_valid(date0);
  ensure_valid(date1);
  return (int) date0.get_offset() - date1.get_offset();
}


template<class DATE> inline DATE operator+(DATE const date, int const shift)
  { return days_after(date, shift); }
template<class DATE> inline DATE operator-(DATE const date, int const shift)
  { return days_before(date, shift); }
template<class DATE> inline int operator-(DATE const date0, DATE const date1)
  { return days_between(date0, date1); } 

template<class DATE> inline DATE operator+=(DATE& date, int const days) 
  { return date = date + days; }
template<class DATE> inline DATE operator++(DATE& date) 
  { return date = date + 1; }
template<class DATE> inline DATE operator++(DATE& date, int /* tag */) 
  { auto old = date; date = date + 1; return old; }
template<class DATE> inline DATE operator-=(DATE& date, int const days) 
  { return date = date -days; }
template<class DATE> inline DATE operator--(DATE& date) 
  { return date = date - 1; }
template<class DATE> inline DATE operator--(DATE& date, int /* tag */) 
  { auto old = date; date = date  -1; return old; }

//------------------------------------------------------------------------------

}  // namespace date
}  // namespace cron

