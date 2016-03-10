#pragma once

#include <cmath>
#include <iostream>

#include "cron/format.hh"
#include "cron/time_zone.hh"
#include "py.hh"

namespace alxs {

using namespace py;

using std::make_unique;
using std::string;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

extern StructSequenceType* get_time_zone_parts_type();

cron::TimeZone_ptr to_time_zone(Object*);
cron::TimeZone_ptr convert_to_time_zone(Object*);

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
  static ref<PyTimeZone> create(cron::TimeZone_ptr tz, PyTypeObject* type=&type_);

  /**
   * Returns true if 'object' is an instance of this type.
   */
  static bool Check(PyObject* object);

  PyTimeZone(cron::TimeZone_ptr tz) : tz_(tz) {}

  /**
   * The wrapped date instance.
   *
   * This is the only non-static data member.
   */
  cron::TimeZone_ptr const tz_;

  // Number methods.
  static PyNumberMethods tp_as_number_;

  // Methods.
  static ref<Object> method_convert (PyTypeObject*, Tuple*, Dict*);
  static ref<Object> method_get     (PyTypeObject*, Tuple*, Dict*);
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


// FIXME: Use a singleton object per underlying time zone.

inline ref<PyTimeZone>
PyTimeZone::create(
  cron::TimeZone_ptr const tz,
  PyTypeObject* const type)
{
  auto self = ref<PyTimeZone>::take(
    check_not_null(PyTimeZone::type_.tp_alloc(type, 0)));

  // tz_ is const to indicate immutablity, but Python initialization is later
  // than C++ initialization, so we have to cast off const here.
  const_cast<cron::TimeZone_ptr&>(self->tz_) = tz;
  return self;
}


inline bool
PyTimeZone::Check(
  PyObject* const other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


//------------------------------------------------------------------------------

}  // namespace alxs

