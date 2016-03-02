#pragma once

#include "PyDate.hh"
#include "PyDaytime.hh"
#include "PyTimeZone.hh"

using namespace alxs;
using namespace py;

namespace alxs {

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

using DefaultDate = PyDate<cron::Date>;
using DefaultDaytime = PyDaytime<cron::Daytime>;

inline ref<Object>
make_date(
  cron::Datenum const datenum,
  Object* type=(Object*) &DefaultDate::type_)
{
  // Special case fast path for the default date type.
  if (type == (Object*) &DefaultDate::type_)
    return DefaultDate::create(DefaultDate::Date::from_datenum(datenum));
  else
    // FIXME
    assert(false);
}


inline ref<Object>
make_daytime(
  cron::Daytick const daytick,
  Object* type=(Object*) &DefaultDaytime::type_)
{
  // Special case fast path for the default daytime type.
  if (type == (Object*) &DefaultDaytime::type_)
    return DefaultDaytime::create(
      DefaultDaytime::Daytime::from_daytick(daytick));
  else
    // FIXME
    assert(false);
}


inline cron::Datenum
to_datenum(
  Object* const obj)
{
  // If the date looks like a long, interpret it as a datenum.
  auto datenum_val = obj->maybe_long_value();
  if (datenum_val && cron::datenum_is_valid(*datenum_val))
    return *datenum_val;

  // Otherwise, look for a datenum attribute or property.
  auto datenum_attr = obj->maybe_get_attr("datenum");
  if (datenum_attr) 
    return (*datenum_attr)->long_value();

  throw Exception(PyExc_TypeError, "not a date or datenum");
}


inline cron::Daytick
to_daytick(
  Object* const obj)
{
  // If the time looks like a number, interpret it as SSM.
  auto ssm_val = obj->maybe_double_value();
  if (ssm_val && cron::ssm_is_valid(*ssm_val))
    return cron::ssm_to_daytick(*ssm_val);

  // Otherwise, look for a daytick attribute or property.
  auto daytick_attr = obj->maybe_get_attr("daytick");
  if (daytick_attr)
    return (*daytick_attr)->long_value();

  throw Exception(PyExc_TypeError, "not a time or SSM");
}


// FIXME: Accept pytz time zones.
inline cron::TimeZone const&
to_time_zone(
  Object* const arg)
{
  if (!PyTimeZone::Check(arg))
    throw Exception(PyExc_TypeError, "tz not a TimeZone");
  return *cast<PyTimeZone>(arg)->tz_;
}


//------------------------------------------------------------------------------

}  // namespace alxs

