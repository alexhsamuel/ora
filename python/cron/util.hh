#pragma once

#include <Python.h>

#include "cron/types.hh"
#include "py.hh"
#include "PyDate.hh"
#include "PyDaytime.hh"

using namespace py;

namespace alxs {

//------------------------------------------------------------------------------
// Declarations

StructSequenceType* get_local_time_type();

//------------------------------------------------------------------------------
// Helpers

inline ref<Object>
make_local(
  cron::LocalDatenumDaytick const local,
  Object* date_type=(Object*) &PyDateDefault::type_,
  Object* daytime_type=(Object*) &PyDaytimeDefault::type_)
{
  auto result = get_local_time_type()->New();
  result->initialize(0, make_date(local.datenum, date_type));
  result->initialize(1, make_daytime(local.daytick, daytime_type));
  return std::move(result);
}


inline ref<Object>
make_local_datenum_daytick(
  cron::LocalDatenumDaytick const local)
{
  auto result = get_local_time_type()->New();
  result->initialize(0, Long::FromLong(local.datenum));
  result->initialize(1, Long::FromUnsignedLong(local.daytick));
  return std::move(result);
}


inline cron::Datenum
to_datenum(
  Object* const obj)
{
  // If the date looks like a long, interpret it as a datenum.
  auto datenum_val = obj->maybe_long_value();
  if (datenum_val && cron::datenum_is_valid(*datenum_val))
    return *datenum_val;

  // FIXME: Use API, or convert to date and then.

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

  // FIXME: Use API, or convert to daytime and then.

  // Otherwise, look for a daytick attribute or property.
  auto daytick = obj->maybe_get_attr("daytick");
  if (daytick)
    return (*daytick)->unsigned_long_value();

  throw Exception(PyExc_TypeError, "not a time or SSM");
}


//------------------------------------------------------------------------------

}  // namespace alxs


