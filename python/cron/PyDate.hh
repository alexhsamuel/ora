#pragma once

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include <Python.h>

#include "cron/date.hh"
#include "cron/format.hh"
#include "py.hh"

using namespace alxs;
using namespace py;

using std::string;
using std::make_unique;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

// FIXME: Should we cache parts?

template<typename TRAITS>
class PyDate
  : public ExtensionType
{
public:

  using Date = cron::DateTemplate<TRAITS>;

  /** 
   * Readies the Python type and adds it to `module` as `name`.  
   *
   * Should only be called once; this is not checked.
   */
  static void add_to(Module& module, string const& name);

  static ref<PyDate> create(Date date);

  Date const date_;

private:

  static int tp_init(PyDate* self, Tuple* args, Dict* kw_args);
  static void tp_dealloc(PyDate* self);
  static Unicode* tp_repr(PyDate* self);
  static Unicode* tp_str(PyDate* self);

  // Singleton objects, constructed lazily.
  static ref<PyDate> INVALID_;
  static ref<PyDate> LAST_;
  static ref<PyDate> MAX_;
  static ref<PyDate> MIN_;
  static ref<PyDate> MISSING_;

  static ref<Object> get_datenum    (PyDate* self, void*);
  static ref<Object> get_day        (PyDate* self, void*);
  static ref<Object> get_invalid    (PyDate* self, void*);
  static ref<Object> get_missing    (PyDate* self, void*);
  static ref<Object> get_month      (PyDate* self, void*);
  static ref<Object> get_ordinal    (PyDate* self, void*);
  static ref<Object> get_valid      (PyDate* self, void*);
  static ref<Object> get_week       (PyDate* self, void*);
  static ref<Object> get_week_year  (PyDate* self, void*);
  static ref<Object> get_weekday    (PyDate* self, void*);
  static ref<Object> get_year       (PyDate* self, void*);
  static GetSets<PyDate> getsets_;

  static Type build_type(string const& type_name);

  /** Date format used to generate the repr.  */
  static unique_ptr<cron::DateFormat> repr_format_;

public:

  static Type type_;

};


template<typename TRAITS>
void
PyDate<TRAITS>::add_to(
  Module& module,
  string const& name)
{
  // Construct the type struct.
  type_ = build_type(string{module.GetName()} + "." + name);
  // Hand it to Python.
  type_.Ready();

  // Build the repr format.
  repr_format_ = make_unique<cron::DateFormat>(
    name + "(%Y, %m, %d)",
    name + ".INVALID",
    name + ".MISSING");

  // Add in static data members.
  Dict* dict = (Dict*) type_.tp_dict;
  assert(dict != nullptr);
  INVALID_  = create(Date::INVALID);
  LAST_     = create(Date::LAST);
  MAX_      = create(Date::MAX);
  MIN_      = create(Date::MIN);
  MISSING_  = create(Date::MISSING);
  dict->SetItemString("INVALID",    INVALID_);
  dict->SetItemString("LAST",       LAST_);
  dict->SetItemString("MAX",        MAX_);
  dict->SetItemString("MIN",        MIN_);
  dict->SetItemString("MISSING",    MISSING_);

  // Add the type to the module.
  module.add(&type_);
}


template<typename TRAITS>
ref<PyDate<TRAITS>>
PyDate<TRAITS>::create(
  Date date)
{
  // FIXME: Check for nullptr?  Or wrap tp_alloc?
  auto obj = ref<PyDate>::take(PyDate::type_.tp_alloc(&PyDate::type_, 0));

  // date_ is const to indicate immutablity, but Python initialization is later
  // than C++ initialization, so we have to cast off const here.
  new(const_cast<Date*>(&obj->date_)) Date{date};
  return obj;
}


//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

// FIXME: Wrap tp_init.
template<typename TRAITS>
int
PyDate<TRAITS>::tp_init(
  PyDate* self, 
  Tuple* args, 
  Dict* kw_args)
{
  static char const* arg_names[] = {"year", "month", "day", nullptr};

  unsigned short year;
  unsigned short month;
  unsigned short day;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "HHH", arg_names, &year, &month, &day);

  try {
    // date_ is const to indicate immutablity, but Python initialization is
    // later than C++ initialization, so we have to cast off const here.
    new(const_cast<Date*>(&self->date_))
      Date{(cron::Year) year, (cron::Month) (month - 1), (cron::Day) (day - 1)};
  }
  catch (cron::DateError error) {
    throw new py::ValueError(error.what());
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


// FIXME: Wrap tp_repr.
template<typename TRAITS>
Unicode*
PyDate<TRAITS>::tp_repr(
  PyDate* self)
{
  return Unicode::from((*repr_format_)(self->date_)).release();
}


// FIXME: Wrap tp_str.
template<typename TRAITS>
Unicode*
PyDate<TRAITS>::tp_str(
  PyDate* self)
{
  // FIXME: Make the format configurable.
  auto& format = cron::DateFormat::get_default();
  return Unicode::from(format(self->date_)).release();
}


//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

template<typename TRAITS>
ref<PyDate<TRAITS>>
PyDate<TRAITS>::INVALID_;


template<typename TRAITS>
ref<PyDate<TRAITS>>
PyDate<TRAITS>::LAST_;


template<typename TRAITS>
ref<PyDate<TRAITS>>
PyDate<TRAITS>::MAX_;


template<typename TRAITS>
ref<PyDate<TRAITS>>
PyDate<TRAITS>::MIN_;


template<typename TRAITS>
ref<PyDate<TRAITS>>
PyDate<TRAITS>::MISSING_;


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_datenum(
  PyDate* self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_datenum());
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_day(
  PyDate* self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().day + 1);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_invalid(
  PyDate* self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_invalid());
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_missing(
  PyDate* self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_missing());
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_month(
  PyDate* self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().month + 1);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_ordinal(
  PyDate* self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().ordinal);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_valid(
  PyDate* self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_valid());
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_week(
  PyDate* self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().week);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_week_year(
  PyDate* self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().week_year);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_weekday(
  PyDate* self,
  void* /* closure */)
{
  // FIXME: Use an enum.
  return Long::FromLong(self->date_.get_parts().weekday);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_year(
  PyDate* self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().year);
}


template<typename TRAITS>
GetSets<PyDate<TRAITS>>
PyDate<TRAITS>::getsets_ 
  = GetSets<PyDate>()
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
Type
PyDate<TRAITS>::build_type(
  string const& type_name)
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
    (reprfunc)            tp_repr,                        // tp_repr
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
unique_ptr<cron::DateFormat>
PyDate<TRAITS>::repr_format_;


template<typename TRAITS>
Type
PyDate<TRAITS>::type_;


// FIXME: API:
//   parts
//   __eq__ & co.
//   copy ctor
//   conversion from other dates
//   from_datenum()
//   ctor from ymd triplet
//   sloppy ctor

