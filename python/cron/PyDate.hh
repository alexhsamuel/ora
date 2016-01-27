#pragma once

#include <cstring>
#include <iostream>

#include <Python.h>

#include "cron/date.hh"
#include "cron/format.hh"
#include "py.hh"

using namespace alxs;

//------------------------------------------------------------------------------

template<typename TRAITS>
class PyDate
  : public py::ExtensionType
{
public:

  static py::Type type_;
  static py::Type build_type(char const* name);

  using Date = cron::DateTemplate<TRAITS>;

  Date date_;

private:

  static int tp_init(PyDate* self, py::Tuple* args, py::Dict* kw_args);
  static void tp_dealloc(PyDate* self);
  static py::Unicode* tp_str(PyDate* self);

};


//------------------------------------------------------------------------------

// FIXME: Wrap tp_init.
template<typename TRAITS>
int
PyDate<TRAITS>::tp_init(
  PyDate* self, 
  py::Tuple* args, 
  py::Dict* kw_args)
{
  static char const* arg_names[] = {"year", "month", "day", nullptr};

  unsigned short year;
  unsigned short month;
  unsigned short day;
  py::Arg::ParseTupleAndKeywords(
    args, kw_args, "HHH", arg_names, &year, &month, &day);

  try {
    new(&self->date_) 
      Date{(cron::Year) year, (cron::Month) (month - 1), (cron::Day) (day - 1)};
  }
  catch (cron::DateError error) {
    throw new ValueError(error.what());
  }

  return 0;
}


// FIXME: Wrap tp_dealloc.
template<typename TRAITS>
void
PyDate<TRAITS>::tp_dealloc(PyDate* self)
{
  self->date_.~DateTemplate();
  self->ob_type->tp_free(self);
}


// FIXME: Wrap tp_str.
template<typename TRAITS>
py::Unicode*
PyDate<TRAITS>::tp_str(
  PyDate* self)
{
  // FIXME: Make the format configurable.
  auto& format = cron::DateFormat::get_default();
  return py::Unicode::from(format(self->date_)).release();
}


template<typename TRAITS>
py::Type
PyDate<TRAITS>::build_type(char const* const name) 
{
  return PyTypeObject{
    PyVarObject_HEAD_INIT(nullptr, 0)
    (char const*)         strdup(name),                   // tp_name
    (Py_ssize_t)          sizeof(PyDate<TRAITS>),         // tp_basicsize
    (Py_ssize_t)          0,                              // tp_itemsize
    (destructor)          tp_dealloc,                     // tp_dealloc
    (printfunc)           nullptr,                        // tp_print
    (getattrfunc)         nullptr,                        // tp_getattr
    (setattrfunc)         nullptr,                        // tp_setattr
    (void*)               nullptr,                        // tp_reserved
    (reprfunc)            nullptr,                        // tp_repr
    (PyNumberMethods*)    nullptr,                        // tp_as_number
    (PySequenceMethods*)  nullptr,                        // tp_as_sequence
    (PyMappingMethods*)   nullptr,                        // tp_as_mapping
    (hashfunc)            nullptr,                        // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            tp_str,                         // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         nullptr,                        // tp_doc
    (traverseproc)        nullptr,                        // tp_traverse
    (inquiry)             nullptr,                        // tp_clear
    (richcmpfunc)         nullptr,                        // tp_richcompare
    (Py_ssize_t)          0,                              // tp_weaklistoffset
    (getiterfunc)         nullptr,                        // tp_iter
    (iternextfunc)        nullptr,                        // tp_iternext
    (PyMethodDef*)        nullptr,                        // tp_methods
    (PyMemberDef*)        nullptr,                        // tp_members
    (PyGetSetDef*)        nullptr,                        // tp_getset
    (_typeobject*)        nullptr,                        // tp_base
    (PyObject*)           nullptr,                        // tp_dict
    (descrgetfunc)        nullptr,                        // tp_descr_get
    (descrsetfunc)        nullptr,                        // tp_descr_set
    (Py_ssize_t)          0,                              // tp_dictoffset
    (initproc)            tp_init,                        // tp_init
    (allocfunc)           nullptr,                        // tp_alloc
    (newfunc)             PyType_GenericNew,              // tp_new
    (freefunc)            nullptr,                        // tp_free
    (inquiry)             nullptr,                        // tp_is_gc
    (PyObject*)           nullptr,                        // tp_bases
    (PyObject*)           nullptr,                        // tp_mro
    (PyObject*)           nullptr,                        // tp_cache
    (PyObject*)           nullptr,                        // tp_subclasses
    (PyObject*)           nullptr,                        // tp_weaklist
    (destructor)          nullptr,                        // tp_del
    (unsigned int)        0,                              // tp_version_tag
    (destructor)          nullptr,                        // tp_finalize
  };
}


template<typename TRAITS>
py::Type
PyDate<TRAITS>::type_
  = PyDate::build_type("cron._ext.Date");  // FIXME


