#pragma once

#include <string>

#include "ora.hh"
#include "py.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

class PyLocalTime
  : public ExtensionType
{
public:

  static void add_to(Module& module, std::string const& name);
  static Type type_;

  static ref<PyLocalTime> create(Object* const date, Object* const daytime, PyTypeObject* typ=&type_);
  static bool Check(PyObject* object);

private:

  PyLocalTime(Object* const date, Object* const daytime);

  ref<Object> const     date_;
  ref<Object> const     daytime_;

  // Getsets.
  static ref<Object>    get_date(PyLocalTime*, void*);
  static ref<Object>    get_daytime(PyLocalTime*, void*);
  static GetSets<PyLocalTime> tp_getsets_;

  static void           tp_dealloc(PyLocalTime*);
  static ref<Unicode>   tp_repr(PyLocalTime*);
  static ref<Unicode>   tp_str(PyLocalTime*);
  static void           tp_init(PyLocalTime*, Tuple*, Dict*);

  static Type build_type(std::string const& type_name);

};


inline ref<PyLocalTime>
PyLocalTime::create(
  Object* const date,
  Object* const daytime,
  PyTypeObject* const type)
{
  auto self = ref<PyLocalTime>::take(
    check_not_null(PyLocalTime::type_.tp_alloc(type, 0)));

  new(self) PyLocalTime(date, daytime);
  return self;
}


inline bool
PyLocalTime::Check(
  PyObject* const other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


inline 
PyLocalTime::PyLocalTime(
  Object* const date,
  Object* const daytime)
: date_(ref<Object>::of(date)),
  daytime_(ref<Object>::of(daytime))
{
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

