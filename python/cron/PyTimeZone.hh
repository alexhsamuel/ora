#pragma once

#include <cmath>
#include <experimental/optional>
#include <iostream>

#include "cron/format.hh"
#include "cron/time_zone.hh"
#include "py.hh"

namespace alxs {

using namespace py;

using std::experimental::optional;
using std::make_unique;
using std::string;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

extern StructSequenceType* get_time_zone_parts_type();

optional<cron::TimeZone const*> convert_time_zone_object(Object*);
optional<cron::TimeZone const*> convert_object_to_time_zone(Object*);

//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

class PyTimeZone
  : public ExtensionType
{
public:

  using TimeZone = cron::TimeZone;

  /** 
   * Readies the Python type and adds it to `module` as `name`.  
   *
   * Should only be called once; this is not checked.
   */
  static void add_to(Module& module, string const& name);

  static Type type_;

  /**
   * Creates an instance of the Python type.
   */
  static ref<PyTimeZone> create(TimeZone const* tz, PyTypeObject* type=&type_);

  /**
   * Returns true if 'object' is an instance of this type.
   */
  static bool Check(PyObject* object);

  PyTimeZone(TimeZone const* tz) : tz_(tz) {}

  /**
   * The wrapped date instance.
   *
   * This is the only non-static data member.
   */
  TimeZone const* const tz_;

  // Number methods.
  static PyNumberMethods tp_as_number_;

  // Methods.
  static ref<Object> method_get(PyTypeObject* type, Tuple* args, Dict* kw_args);
  static Methods<PyTimeZone> tp_methods_;

  // Getsets.
  static ref<Object> get_name(PyTimeZone*, void*);
  static GetSets<PyTimeZone> tp_getsets_;

private:

  static void           tp_dealloc  (PyTimeZone*);
  static ref<Unicode>   tp_str      (PyTimeZone*);
  static ref<Unicode>   tp_repr     (PyTimeZone*);
  static void           tp_init     (PyTimeZone*, Tuple*, Dict*);

  static Type build_type(string const& type_name);

};


inline ref<PyTimeZone>
PyTimeZone::create(
  TimeZone const* const tz,
  PyTypeObject* const type)
{
  auto self = ref<PyTimeZone>::take(
    check_not_null(PyTimeZone::type_.tp_alloc(type, 0)));

  // tz_ is const to indicate immutablity, but Python initialization is later
  // than C++ initialization, so we have to cast off const here.
  *const_cast<TimeZone const**>(&self->tz_) = tz;
  return self;
}


inline bool
PyTimeZone::Check(
  PyObject* const other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


//------------------------------------------------------------------------------
// Helpers

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

