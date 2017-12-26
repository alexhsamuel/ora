#pragma once

#include <Python.h>

#include "ora.hh"
#include "py.hh"
#include "PyDate.hh"
#include "PyDaytime.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------
// Declarations

StructSequenceType* get_local_time_type();

//------------------------------------------------------------------------------
// Helpers

inline ref<Object>
make_local(
  ora::LocalDatenumDaytick const local,
  PyTypeObject* date_type=&PyDateDefault::type_,
  PyTypeObject* daytime_type=&PyDaytimeDefault::type_)
{
  auto result = get_local_time_type()->New();
  result->initialize(0, make_date(local.datenum, date_type));
  result->initialize(1, make_daytime(local.daytick, daytime_type));
  // FIXME: Add time zone info?
  return std::move(result);
}


inline ref<Object>
make_local_datenum_daytick(
  ora::LocalDatenumDaytick const local)
{
  auto result = get_local_time_type()->New();
  result->initialize(0, Long::FromLong(local.datenum));
  result->initialize(1, Long::FromUnsignedLong(local.daytick));
  // FIXME: Add time zone info?
  return std::move(result);
}


inline ora::Datenum
to_datenum(
  Object* const obj)
{
  // If the date looks like a long, interpret it as a datenum.
  auto datenum_val = obj->maybe_long_value();
  if (datenum_val && ora::datenum_is_valid(*datenum_val))
    return *datenum_val;

  // FIXME: Use API, or convert to date and then.

  // Look for a datenum attribute or property.
  auto datenum_attr = obj->maybe_get_attr("datenum");
  if (datenum_attr) 
    return (*datenum_attr)->long_value();

  // Look for a toordinal() method.
  auto toordinal_method = obj->maybe_get_attr("toordinal");
  if (toordinal_method) {
    auto const ordinal_obj = (*toordinal_method)->CallObject(nullptr);
    return ordinal_obj->long_value() - 1;
  }

  throw Exception(PyExc_TypeError, "not a date or datenum");
}


inline ora::Daytick
to_daytick(
  Object* const obj)
{
  // If the time looks like a number, interpret it as SSM.
  auto ssm_val = obj->maybe_double_value();
  if (ssm_val && ora::ssm_is_valid(*ssm_val))
    return ora::ssm_to_daytick(*ssm_val);

  // FIXME: Use API, or convert to daytime and then.

  // Otherwise, look for a daytick attribute or property.
  auto daytick = obj->maybe_get_attr("daytick");
  if (daytick)
    return (*daytick)->unsigned_long_value();

  throw Exception(PyExc_TypeError, "not a time or SSM");
}


inline std::pair<ora::Datenum, ora::Daytick>
to_datenum_daytick(
  Object* const obj)
{
  // FIXME: Check for a LocalTime object.

  // A (date, daytime) sequence.
  if (Sequence::Check(obj)) {
    auto const seq = cast<Sequence>(obj);
    if (seq->Length() == 2)
      return {to_datenum(seq->GetItem(0)), to_daytick(seq->GetItem(1))};
  }

  // FIXME: Check for naive datetime.datetime.

  throw TypeError("not a localtime: "s + *obj->Repr());
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

