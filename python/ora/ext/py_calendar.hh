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

  using Cal_ptr = std::shared_ptr<Calendar>;

  static Type type_;
  static Type build_type();
  static void add_to(Module& module);

  static ref<PyCalendar>
  create(
    Cal_ptr&& cal,
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

  PyCalendar(Cal_ptr const& cal) : cal_(cal) {}
  PyCalendar(Cal_ptr&& cal) : cal_(std::move(cal)) {}

  Cal_ptr cal_;

};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

