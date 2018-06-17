#pragma once

#include <Python.h>

#include "ora.hh"
#include "py.hh"
#include "py_date.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

inline Interval<Date>
parse_range(
  Object* arg)
{
  if (Sequence::Check(arg)) {
    auto seq = cast<Sequence>(arg);
    if (seq->Length() == 2) {
      auto min = convert_to_date(seq->GetItem(0));
      auto max = convert_to_date(seq->GetItem(1));
      if (min <= max)
        return {min, max};
      else
        throw ValueError("range max cannot precede min");
    }
  }

  throw TypeError("not a date range");
}


//------------------------------------------------------------------------------

class PyCalendar
: public ExtensionType
{
public:

  static Type type_;
  static Type build_type();
  static void add_to(Module& module);

  static ref<PyCalendar>
  create(
    Calendar&& cal,
    PyTypeObject* type=&type_)
  {
    auto self = ref<PyCalendar>::take(
      check_not_null(PyCalendar::type_.tp_alloc(type, 0)));
    new(self) PyCalendar(std::move(cal));
    return self;
  }

  static bool 
  Check(
    PyObject* object)
  {
    return static_cast<Object*>(object)->IsInstance((PyObject*) &type_);
  }

  PyCalendar(
    Calendar&& cal, 
    Object* const name=nullptr) 
  : cal_(std::move(cal))
  , name_(name == nullptr ? ref<Unicode>() : name->Str())
  {
  }

  Calendar const cal_;
  ref<Unicode> name_;

};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

