#pragma once

#include <iostream>  // FIXME: Remove.
#include <limits>
#include <string>

#include "aslib/exc.hh"
#include "aslib/math.hh"
#include "cron/date_math.hh"
#include "cron/date_functions.hh"
#include "cron/date_safe.hh"
#include "cron/date_tmpl.hh"
#include "cron/types.hh"

namespace cron {
namespace date {

//------------------------------------------------------------------------------

namespace {

template<class TRAITS>
void ensure_valid(
  DateTemplate<TRAITS> const date)
{
  if (!date.is_valid())
    throw InvalidDateError();
}


}  // anonymous namespace

//------------------------------------------------------------------------------
// Day arithmetic
//------------------------------------------------------------------------------

template<class TRAITS> 
extern inline DateTemplate<TRAITS> 
operator+(
  DateTemplate<TRAITS> date, 
  int shift)
{
  ensure_valid(date);
  return DateTemplate<TRAITS>::from_offset(date.get_offset() + shift);
}


template<class TRAITS> 
extern inline DateTemplate<TRAITS> 
operator-(
  DateTemplate<TRAITS> date, 
  int shift)
{
  ensure_valid(date);
  return DateTemplate<TRAITS>::from_offset(date.get_offset() - shift);
}


template<class TRAITS>
extern inline int
operator-(
  DateTemplate<TRAITS> date0,
  DateTemplate<TRAITS> date1)
{
  ensure_valid(date0);
  ensure_valid(date1);
  return (int) date0.get_offset() - date1.get_offset();
}


template<class TRAITS> DateTemplate<TRAITS> operator+=(DateTemplate<TRAITS>& date, ssize_t days) { return date = date + days; }
template<class TRAITS> DateTemplate<TRAITS> operator++(DateTemplate<TRAITS>& date) { return date = date + 1; }
template<class TRAITS> DateTemplate<TRAITS> operator++(DateTemplate<TRAITS>& date, int) { auto old = date; date = date + 1; return old; }
template<class TRAITS> DateTemplate<TRAITS> operator-=(DateTemplate<TRAITS>& date, ssize_t days) { return date = date -days; }
template<class TRAITS> DateTemplate<TRAITS> operator--(DateTemplate<TRAITS>& date) { return date = date - 1; }
template<class TRAITS> DateTemplate<TRAITS> operator--(DateTemplate<TRAITS>& date, int) { auto old = date; date = date  -1; return old; }

//------------------------------------------------------------------------------

}  // namespace date
}  // namespace cron


