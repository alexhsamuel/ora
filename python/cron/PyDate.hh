#pragma once

#include <cstring>
#include <iostream>

#include <Python.h>

#include "cron/date.hh"
#include "cron/format.hh"
#include "py.hh"

using namespace alxs;

//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

template<typename TRAITS>
class PyDate
  : public py::ExtensionType
{
public:

  static void add_to(py::Module& module, std::string const& name);

  using Date = cron::DateTemplate<TRAITS>;

  Date const date_;

private:

  static int tp_init(PyDate* self, py::Tuple* args, py::Dict* kw_args);
  static void tp_dealloc(PyDate* self);
  static py::Unicode* tp_str(PyDate* self);

  static py::ref<py::Object> get_datenum    (PyDate* self, void*);
  static py::ref<py::Object> get_day        (PyDate* self, void*);
  static py::ref<py::Object> get_invalid    (PyDate* self, void*);
  static py::ref<py::Object> get_missing    (PyDate* self, void*);
  static py::ref<py::Object> get_month      (PyDate* self, void*);
  static py::ref<py::Object> get_ordinal    (PyDate* self, void*);
  static py::ref<py::Object> get_valid      (PyDate* self, void*);
  static py::ref<py::Object> get_week       (PyDate* self, void*);
  static py::ref<py::Object> get_week_year  (PyDate* self, void*);
  static py::ref<py::Object> get_weekday    (PyDate* self, void*);
  static py::ref<py::Object> get_year       (PyDate* self, void*);
  static py::GetSets<PyDate> getsets_;

  static py::Type build_type(std::string const& type_name);

public:

  static py::Type type_;

};


template<typename TRAITS>
void
PyDate<TRAITS>::add_to(
  py::Module& module,
  std::string const& name)
{
  type_ = build_type(std::string{module.GetName()} + "." + name);
  type_.Ready();
  module.add(&type_);
}


//------------------------------------------------------------------------------
// Standard type methods
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
    // date_ is const to indicate immutable state, but Python initialization
    // is later than C++ initialization, so we have to cast off const here.
    new(const_cast<Date*>(&self->date_))
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


//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_datenum(
  PyDate* self,
  void* /* closure */)
{
  return py::Long::FromLong(self->date_.get_datenum());
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_day(
  PyDate* self,
  void* /* closure */)
{
  return py::Long::FromLong(self->date_.get_parts().day + 1);
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_invalid(
  PyDate* self,
  void* /* closure */)
{
  return py::Bool::from(self->date_.is_invalid());
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_missing(
  PyDate* self,
  void* /* closure */)
{
  return py::Bool::from(self->date_.is_missing());
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_month(
  PyDate* self,
  void* /* closure */)
{
  return py::Long::FromLong(self->date_.get_parts().month + 1);
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_ordinal(
  PyDate* self,
  void* /* closure */)
{
  return py::Long::FromLong(self->date_.get_parts().ordinal);
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_valid(
  PyDate* self,
  void* /* closure */)
{
  return py::Bool::from(self->date_.is_valid());
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_week(
  PyDate* self,
  void* /* closure */)
{
  return py::Long::FromLong(self->date_.get_parts().week);
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_week_year(
  PyDate* self,
  void* /* closure */)
{
  return py::Long::FromLong(self->date_.get_parts().week_year);
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_weekday(
  PyDate* self,
  void* /* closure */)
{
  // FIXME: Use an enum.
  return py::Long::FromLong(self->date_.get_parts().weekday);
}


template<typename TRAITS>
py::ref<py::Object>
PyDate<TRAITS>::get_year(
  PyDate* self,
  void* /* closure */)
{
  return py::Long::FromLong(self->date_.get_parts().year);
}


template<typename TRAITS>
py::GetSets<PyDate<TRAITS>>
PyDate<TRAITS>::getsets_ 
  = py::GetSets<PyDate>()
    .template add_get<get_datenum>      ("datenum")
    .template add_get<get_day>          ("day")
    .template add_get<get_invalid>      ("invalid")
    .template add_get<get_missing>      ("missing")
    .template add_get<get_month>        ("month")
    .template add_get<get_ordinal>      ("ordinal")
    .template add_get<get_valid>        ("valid")
    .template add_get<get_week>         ("week")
    .template add_get<get_week_year>    ("week_year")
    .template add_get<get_weekday>      ("weekday")
    .template add_get<get_year>         ("year")
  ;


//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

template<typename TRAITS>
py::Type
PyDate<TRAITS>::build_type(
  std::string const& type_name)
{
  return PyTypeObject{
    PyVarObject_HEAD_INIT(nullptr, 0)
    (char const*)         strdup(type_name.c_str()),      // tp_name
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
    (PyGetSetDef*)        getsets_,                       // tp_getset
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
PyDate<TRAITS>::type_;


// FIXME: API:
//   parts
//   __eq__ & co.
//   copy ctor
//   conversion from other dates
//   from_datenum()
//   ctor from ymd triplet
//   sloppy ctor
//   MIN
//   LAST
//   MAX
//   INVALID
//   MISSING

