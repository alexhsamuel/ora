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

  static ref<PyLocal> create(Object* const date, Object* const daytime, PyTypeObject* typ=&type_);
  static bool Check(PyObject* object);

private:

  PyLocal(Object* const date, Object* const daytime);

  ref<Object> const     date_;
  ref<Object> const     daytime_;

  // Getsets.
  static ref<Object>    get_date(PyLocal*, void*);
  static ref<Object>    get_daytime(PyLocal*, void*);
  static GetSets<PyLocal> tp_getsets_;

  static void           tp_dealloc(PyLocal*);
  static ref<Unicode>   tp_repr(PyLocal*);
  static ref<Unicode>   tp_str(PyLocal*);
  static ref<Object>    tp_richcompare(PyLocal*, Object*, int);
  static void           tp_init(PyLocal*, Tuple*, Dict*);

  static Py_ssize_t     sq_length(PyLocal*);
  static ref<Object>    sq_item(PyLocal*, Py_ssize_t);
  static PySequenceMethods const tp_as_sequence;

  static Type build_type(std::string const& type_name);

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

