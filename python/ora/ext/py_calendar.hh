#pragma once

#include <Python.h>

#include "ora.hh"
#include "py.hh"
#include "py_date.hh"

namespace ora {
namespace py {

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

  PyCalendar(Calendar&& cal) : cal_(std::move(cal)) {}

  Calendar const cal_;

};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

