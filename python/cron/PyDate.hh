#pragma once

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include <Python.h>

#include "cron/date.hh"
#include "cron/format.hh"
#include "py.hh"

namespace alxs {

using namespace py;

using std::string;
using std::make_unique;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Parts type
//------------------------------------------------------------------------------

StructSequenceType*
get_date_parts_type();

ref<Object>
get_month_obj(int month);

ref<Object>
get_weekday_obj(int weekday);

//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

// FIXME: Think carefully over when to return INVALID versus when to raise.

// FIXME: Should we cache parts?

// FIXME: Template argument should be DATE, not TRAITS.
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

  static ref<PyDate> create(Date date, PyTypeObject* type=&type_);

  static bool Check(PyObject* object);

  Date const date_;

  PyDate(Date date) : date_(date) {}

private:

  static void tp_init(PyDate* self, Tuple* args, Dict* kw_args);
  static void tp_dealloc(PyDate* self);
  static ref<Unicode> tp_repr(PyDate* self);
  static ref<Unicode> tp_str(PyDate* self);
  static Object* tp_richcompare(PyDate* self, Object* other, int comparison);

  // Singleton objects, constructed lazily.
  static ref<PyDate> INVALID_;
  static ref<PyDate> LAST_;
  static ref<PyDate> MAX_;
  static ref<PyDate> MIN_;
  static ref<PyDate> MISSING_;

  // Methods.
  static ref<Object> method_from_datenum(PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_ordinal(PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_parts(PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_ymdi(PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_is_same(PyDate* self, Tuple* args, Dict* kw_args);
  static Methods<PyDate> tp_methods_;

  // Getsets.
  static ref<Object> get_datenum    (PyDate* self, void*);
  static ref<Object> get_day        (PyDate* self, void*);
  static ref<Object> get_invalid    (PyDate* self, void*);
  static ref<Object> get_missing    (PyDate* self, void*);
  static ref<Object> get_month      (PyDate* self, void*);
  static ref<Object> get_ordinal    (PyDate* self, void*);
  static ref<Object> get_parts      (PyDate* self, void*);
  static ref<Object> get_valid      (PyDate* self, void*);
  static ref<Object> get_week       (PyDate* self, void*);
  static ref<Object> get_week_year  (PyDate* self, void*);
  static ref<Object> get_weekday    (PyDate* self, void*);
  static ref<Object> get_year       (PyDate* self, void*);
  static ref<Object> get_ymdi       (PyDate* self, void*);
  static GetSets<PyDate> tp_getsets_;

  /** Date format used to generate the repr.  */
  static unique_ptr<cron::DateFormat> repr_format_;

  static Type build_type(string const& type_name);

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
    name + "(%0Y, %0m, %0d)",
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
  Date date,
  PyTypeObject* type)
{
  // FIXME: Check for nullptr?  Or wrap tp_alloc?
  auto obj = ref<PyDate>::take(PyDate::type_.tp_alloc(type, 0));

  // date_ is const to indicate immutablity, but Python initialization is later
  // than C++ initialization, so we have to cast off const here.
  new(const_cast<Date*>(&obj->date_)) Date{date};
  return obj;
}


template<typename TRAITS>
bool
PyDate<TRAITS>::Check(
  PyObject* other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

template<typename TRAITS>
void
PyDate<TRAITS>::tp_init(
  PyDate* const self, 
  Tuple* args, 
  Dict* kw_args)
{
  Object* obj = nullptr;
  Arg::ParseTuple(args, "|O", &obj);

  Date date;

  if (obj == nullptr) 
    // Use the default value.
    ;

  else if (PyDate::Check(obj)) 
    // Same type.
    date = static_cast<PyDate*>(obj)->date_;

  // FIXME: Check for other PyDate types?

  else {
    // Try for a date type that has a 'datenum' attribute.
    auto datenum = obj->GetAttrString("datenum", false);
    if (datenum != nullptr) 
      date = Date::from_datenum(datenum->long_value());

    else {
      // Try for a date type that as a 'toordinal()' method.
      auto ordinal = obj->CallMethodObjArgs("toordinal", false);
      if (ordinal != nullptr)
        date = Date::from_datenum(ordinal->long_value() - 437986);

      else 
        // No type match.
        throw TypeError("not a date");
    }
  }

  new(self) PyDate{date};
}


// FIXME: Wrap tp_dealloc.
template<typename TRAITS>
void
PyDate<TRAITS>::tp_dealloc(PyDate* const self)
{
  self->date_.~DateTemplate();
  self->ob_type->tp_free(self);
}


template<typename TRAITS>
ref<Unicode>
PyDate<TRAITS>::tp_repr(
  PyDate* const self)
{
  return Unicode::from((*repr_format_)(self->date_));
}


template<typename TRAITS>
ref<Unicode>
PyDate<TRAITS>::tp_str(
  PyDate* const self)
{
  // FIXME: Make the format configurable.
  auto& format = cron::DateFormat::get_default();
  return Unicode::from(format(self->date_));
}


// FIXME: Wrap tp_richcompare.
template<typename TRAITS>
Object*
PyDate<TRAITS>::tp_richcompare(
  PyDate* const self,
  Object* const other,
  int const comparison)
{
  // FIXME: Allow comparison to other date types.
  if (! PyDate::Check(other))
    return not_implemented_ref().release();

  Date const& d0 = self->date_;
  Date const& d1 = ((PyDate*) other)->date_;

  bool result;
  switch (comparison) {
  case Py_EQ: result = d0 == d1; break;
  case Py_GE: result = d0 >= d1; break;
  case Py_GT: result = d0 >  d1; break;
  case Py_LE: result = d0 <= d1; break;
  case Py_LT: result = d0 <  d1; break;
  case Py_NE: result = d0 != d1; break;
  default:    result = false; assert(false);
  }
  return Bool::from(result).release();
}


//------------------------------------------------------------------------------
// Methods
//------------------------------------------------------------------------------

template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::method_from_datenum(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"datenum", nullptr};
  cron::Datenum datenum;
  static_assert(
    sizeof(cron::Datenum) == sizeof(int),
    "datenum is not an int");
  Arg::ParseTupleAndKeywords(args, kw_args, "i", arg_names, &datenum);

  return create(Date::from_datenum(datenum), type);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::method_from_ordinal(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"year", "ordinal", nullptr};
  cron::Year year;
  cron::Ordinal ordinal;
  static_assert(sizeof(cron::Year) == sizeof(short), "year is not a short");
  static_assert(sizeof(cron::Ordinal) == sizeof(short), "ordinal is not a short");
  Arg::ParseTupleAndKeywords(args, kw_args, "HH", arg_names, &year, &ordinal);

  return create(Date::from_ordinal(year, ordinal), type);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::method_from_parts(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("from_parts() takes no keyword arguments");

  Sequence* parts;
  // Accept either a single three-element sequence, or three args.
  if (args->Length() == 1) {
    parts = cast<Sequence>(args->GetItem(0));
    if (parts->Length() < 3)
      throw TypeError("parts must be a 3-element or longer sequence");
  }
  else if (args->Length() == 3)
    parts = args;
  else
    throw TypeError("from_parts() takes one or three arguments");

  long const year   = parts->GetItem(0)->long_value();
  long const month  = parts->GetItem(1)->long_value();
  long const day    = parts->GetItem(2)->long_value();
  return create(Date(year, month - 1, day - 1), type);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::method_from_ymdi(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"ymdi", nullptr};
  int ymdi;
  Arg::ParseTupleAndKeywords(args, kw_args, "i", arg_names, &ymdi);

  return create(Date::from_ymdi(ymdi), type);
}


// We call this method "is_same" because "is" is a keyword in Python.
template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::method_is_same(
  PyDate* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"object", nullptr};
  Object* object;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &object);

  return Bool::from(
       PyDate::Check(object) 
    && self->date_.is(reinterpret_cast<PyDate const*>(object)->date_));
}


template<typename TRAITS>
Methods<PyDate<TRAITS>>
PyDate<TRAITS>::tp_methods_
  = Methods<PyDate>()
    .template add_class<method_from_datenum>        ("from_datenum")
    .template add_class<method_from_ordinal>        ("from_ordinal")
    .template add_class<method_from_parts>          ("from_parts")
    .template add_class<method_from_ymdi>           ("from_ymdi")
    .template add<method_is_same>                   ("is_same")
  ;


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
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_datenum());
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_day(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().day + 1);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_invalid(
  PyDate* const self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_invalid());
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_missing(
  PyDate* const self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_missing());
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_month(
  PyDate* const self,
  void* /* closure */)
{
  return get_month_obj(self->date_.get_parts().month + 1);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_ordinal(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().ordinal + 1);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_parts(
  PyDate* const self,
  void* /* closure */)
{
  auto parts = self->date_.get_parts();
  auto parts_obj = get_date_parts_type()->New();
  parts_obj->initialize(0, Long::FromLong(parts.year));
  parts_obj->initialize(1, get_month_obj(parts.month + 1));
  parts_obj->initialize(2, Long::FromLong(parts.day + 1));
  parts_obj->initialize(3, Long::FromLong(parts.ordinal + 1));
  parts_obj->initialize(4, Long::FromLong(parts.week_year));
  parts_obj->initialize(5, Long::FromLong(parts.week));
  parts_obj->initialize(6, get_weekday_obj(parts.weekday));
  return std::move(parts_obj);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_valid(
  PyDate* const self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_valid());
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_week(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().week);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_week_year(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().week_year);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_weekday(
  PyDate* const self,
  void* /* closure */)
{
  return get_weekday_obj(self->date_.get_parts().weekday);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_year(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().year);
}


template<typename TRAITS>
ref<Object>
PyDate<TRAITS>::get_ymdi(
  PyDate* const self,
  void* /* closure */)
{
  auto const parts = self->date_.get_parts();
  int ymd = 10000 * parts.year + 100 * (parts.month + 1) + (parts.day + 1);
  return Long::FromLong(ymd);
}


template<typename TRAITS>
GetSets<PyDate<TRAITS>>
PyDate<TRAITS>::tp_getsets_ 
  = GetSets<PyDate>()
    .template add_get<get_datenum>      ("datenum")
    .template add_get<get_day>          ("day")
    .template add_get<get_invalid>      ("invalid")
    .template add_get<get_missing>      ("missing")
    .template add_get<get_month>        ("month")
    .template add_get<get_ordinal>      ("ordinal")
    .template add_get<get_parts>        ("parts")
    .template add_get<get_valid>        ("valid")
    .template add_get<get_week>         ("week")
    .template add_get<get_week_year>    ("week_year")
    .template add_get<get_weekday>      ("weekday")
    .template add_get<get_year>         ("year")
    .template add_get<get_ymdi>         ("ymdi")
  ;


//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

template<typename TRAITS>
unique_ptr<cron::DateFormat>
PyDate<TRAITS>::repr_format_;

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
    (reprfunc)            wrap<PyDate, tp_repr>,          // tp_repr
    (PyNumberMethods*)    nullptr,                        // tp_as_number
    (PySequenceMethods*)  nullptr,                        // tp_as_sequence
    (PyMappingMethods*)   nullptr,                        // tp_as_mapping
    (hashfunc)            nullptr,                        // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            wrap<PyDate, tp_str>,           // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         nullptr,                        // tp_doc
    (traverseproc)        nullptr,                        // tp_traverse
    (inquiry)             nullptr,                        // tp_clear
    (richcmpfunc)         tp_richcompare,                 // tp_richcompare
    (Py_ssize_t)          0,                              // tp_weaklistoffset
    (getiterfunc)         nullptr,                        // tp_iter
    (iternextfunc)        nullptr,                        // tp_iternext
    (PyMethodDef*)        tp_methods_,                    // tp_methods
    (PyMemberDef*)        nullptr,                        // tp_members
    (PyGetSetDef*)        tp_getsets_,                    // tp_getset
    (_typeobject*)        nullptr,                        // tp_base
    (PyObject*)           nullptr,                        // tp_dict
    (descrgetfunc)        nullptr,                        // tp_descr_get
    (descrsetfunc)        nullptr,                        // tp_descr_set
    (Py_ssize_t)          0,                              // tp_dictoffset
    (initproc)            wrap<PyDate, tp_init>,          // tp_init
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
Type
PyDate<TRAITS>::type_;


}  // namespace alxs


