#pragma once

#include <string>

#include "ora.hh"
#include "py.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

class PyLocal
  : public ExtensionType
{
public:

  static void add_to(Module& module, std::string const& name);
  static Type type_;

  static Type build_type(std::string const& type_name);
  static ref<PyLocal> create(Object* const date, Object* const daytime, PyTypeObject* typ=&type_);
  static bool Check(PyObject* object);

  PyLocal(Object* const date, Object* const daytime);

  ref<Object> const     date_;
  ref<Object> const     daytime_;

};


inline ref<PyLocal>
PyLocal::create(
  Object* const date,
  Object* const daytime,
  PyTypeObject* const type)
{
  auto self = ref<PyLocal>::take(
    check_not_null(PyLocal::type_.tp_alloc(type, 0)));

  new(self) PyLocal(date, daytime);
  return self;
}


inline bool
PyLocal::Check(
  PyObject* const other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


inline 
PyLocal::PyLocal(
  Object* const date,
  Object* const daytime)
: date_(ref<Object>::of(date)),
  daytime_(ref<Object>::of(daytime))
{
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

