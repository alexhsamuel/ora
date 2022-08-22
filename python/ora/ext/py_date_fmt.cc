#include <iostream>

#include <Python.h>

#include "py.hh"
#include "py_date.hh"
#include "py_date_fmt.hh"

using std::string;

namespace ora {
namespace py {

//------------------------------------------------------------------------------

void
PyDateFmt::add_to(
  Module& module)
{
  type_.Ready();
  module.add(&type_);
}


namespace {

int tp_init(PyDateFmt* self, PyObject* args, PyObject* kw_args)
{
  static char const* arg_names[] = { nullptr };
  if (!PyArg_ParseTupleAndKeywords(args, kw_args, "", (char**) arg_names))
    return -1;

  new(self) PyDateFmt();
  return 0;
}


ref<Unicode> tp_repr(PyDateFmt* self)
{
  return Unicode::from("DateFmt()");
}


ref<Object> tp_call(PyDateFmt* self, Tuple* args, Dict* kw_args)
{
  static char const* arg_names[] = { "value", nullptr };
  Object* arg;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &arg);

  auto const date = convert_to_date(arg);
  return Unicode::from(DateFormat::DEFAULT(date));
}


auto methods = Methods<PyDateFmt>();


ref<Object> get_width(PyDateFmt* const self, void* /* closure */)
{
  return Long::FromLong(10);
}


auto getsets = GetSets<PyDateFmt>()
  .add_get<get_width>       ("width")
  ;


}  // anonymous namespace


Type PyDateFmt::type_ = PyTypeObject{
  PyVarObject_HEAD_INIT(nullptr, 0)
  (char const*)         "DateFmt",                          // tp_name
  (Py_ssize_t)          sizeof(PyDateFmt),                  // tp_basicsize
  (Py_ssize_t)          0,                                  // tp_itemsize
  (destructor)          nullptr,                            // tp_dealloc
  (printfunc)           nullptr,                            // tp_print
  (getattrfunc)         nullptr,                            // tp_getattr
  (setattrfunc)         nullptr,                            // tp_setattr
  (PyAsyncMethods*)     nullptr,                            // tp_as_async
  (reprfunc)            wrap<PyDateFmt, tp_repr>,           // tp_repr
  (PyNumberMethods*)    nullptr,                            // tp_as_number
  (PySequenceMethods*)  nullptr,                            // tp_as_sequence
  (PyMappingMethods*)   nullptr,                            // tp_as_mapping
  (hashfunc)            nullptr,                            // tp_hash
  (ternaryfunc)         wrap<PyDateFmt, tp_call>,           // tp_call
  (reprfunc)            nullptr,                            // tp_str
  (getattrofunc)        nullptr,                            // tp_getattro
  (setattrofunc)        nullptr,                            // tp_setattro
  (PyBufferProcs*)      nullptr,                            // tp_as_buffer
  (unsigned long)       Py_TPFLAGS_DEFAULT
                        | Py_TPFLAGS_BASETYPE,              // tp_flags
  (char const*)         nullptr,                            // tp_doc
  (traverseproc)        nullptr,                            // tp_traverse
  (inquiry)             nullptr,                            // tp_clear
  (richcmpfunc)         nullptr,                            // tp_richcompare
  (Py_ssize_t)          0,                                  // tp_weaklistoffset
  (getiterfunc)         nullptr,                            // tp_iter
  (iternextfunc)        nullptr,                            // tp_iternext
  (PyMethodDef*)        methods,                            // tp_methods
  (PyMemberDef*)        nullptr,                            // tp_members
  (PyGetSetDef*)        getsets,                            // tp_getset
  (_typeobject*)        nullptr,                            // tp_base
  (PyObject*)           nullptr,                            // tp_dict
  (descrgetfunc)        nullptr,                            // tp_descr_get
  (descrsetfunc)        nullptr,                            // tp_descr_set
  (Py_ssize_t)          0,                                  // tp_dictoffset
  (initproc)            tp_init,                            // tp_init
  (allocfunc)           nullptr,                            // tp_alloc
  (newfunc)             PyType_GenericNew,                  // tp_new
  (freefunc)            nullptr,                            // tp_free
  (inquiry)             nullptr,                            // tp_is_gc
  (PyObject*)           nullptr,                            // tp_bases
  (PyObject*)           nullptr,                            // tp_mro
  (PyObject*)           nullptr,                            // tp_cache
  (PyObject*)           nullptr,                            // tp_subclasses
  (PyObject*)           nullptr,                            // tp_weaklist
  (destructor)          nullptr,                            // tp_del
  (unsigned int)        0,                                  // tp_version_tag
  (destructor)          nullptr,                            // tp_finalize
};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

