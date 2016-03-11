#pragma once
#pragma GCC diagnostic ignored "-Wparentheses"

#include <cstring>
#include <experimental/optional>
#include <iostream>
#include <memory>
#include <string>

#include <Python.h>
#include <datetime.h>

#include "cron/date.hh"
#include "cron/format.hh"
#include "py.hh"

namespace alxs {

using namespace py;
using namespace std::literals;

using std::experimental::optional;
using std::make_unique;
using std::string;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

StructSequenceType* get_date_parts_type();

ref<Object> get_month_obj(int month);
ref<Object> get_weekday_obj(int weekday);

/**
 * Attempts to convert various kinds of Python date objects to Date.
 *
 * If 'obj' is a date object of some kind, returns the equivalent date;
 * otherwise a null option.  The following date objects are recognized:
 *
 *  - PyDateTemplate instances
 *  - 'datetime.date' instances
 *  - objects with a 'datenum' attribute
 *  - objects with a 'toordinal()' method
 */
template<typename DATE> optional<DATE> maybe_date(Object*);

/**
 * Converts various kinds of Python objects to Date.
 *
 * If 'obj' can be converted unambiguously to a date, returns it.  Otherwise,
 * raises a Python exception.
 */
template<typename DATE> DATE convert_to_date(Object*);

/**
 * Helper for converting a 3-element sequence of date parts.
 */
template<typename DATE> inline DATE parts_to_date(Sequence*);

/**
 * Helper for converting a 2-element sequence of ordinal date parts.
 */
template<typename DATE> inline DATE ordinal_parts_to_date(Sequence*);

//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

// FIXME: Think carefully over when to return INVALID versus when to raise.

// FIXME: Should we cache parts?

/**
 * Template for a Python extension type wrapping a date class.
 *
 * 'DATE' is the wrapped date class, an instance of DateTemplate.  Invoke
 * add_to() to construct the type's PyTypeObject, ready it, and add it to a
 * module.
 */
template<typename DATE>
class PyDate
  : public ExtensionType
{
public:

  using Date = DATE;

  /** 
   * Readies the Python type and adds it to `module` as `name`.  
   *
   * Should only be called once; this is not checked.
   */
  static void add_to(Module& module, string const& name);

  /**
   * Creates an instance of the Python type.
   */
  static ref<PyDate> create(Date date, PyTypeObject* type=&type_);

  /**
   * Returns true if 'object' is an instance of this type.
   */
  static bool Check(PyObject* object);

  PyDate(Date date) : date_(date) {}

  /**
   * The wrapped date instance.
   *
   * This is the only non-static data member.
   */
  Date const date_;

private:

  static void tp_init(PyDate* self, Tuple* args, Dict* kw_args);
  static void tp_dealloc(PyDate* self);
  static ref<Unicode> tp_repr(PyDate* self);
  static ref<Unicode> tp_str(PyDate* self);
  static ref<Object> tp_richcompare(PyDate* self, Object* other, int comparison);

  // Number methods.
  static ref<Object> nb_add     (PyDate* self, Object* other, bool right);
  static ref<Object> nb_subtract(PyDate* self, Object* other, bool right);
  static PyNumberMethods tp_as_number_;

  // Methods.
  static ref<Object> method_from_datenum        (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_ordinal_date   (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_parts          (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_week_date      (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_ymdi           (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_is_same             (PyDate*       self, Tuple* args, Dict* kw_args);
  static Methods<PyDate> tp_methods_;

  // Getsets.
  static ref<Object> get_datenum                (PyDate* self, void*);
  static ref<Object> get_day                    (PyDate* self, void*);
  static ref<Object> get_invalid                (PyDate* self, void*);
  static ref<Object> get_missing                (PyDate* self, void*);
  static ref<Object> get_month                  (PyDate* self, void*);
  static ref<Object> get_ordinal                (PyDate* self, void*);
  static ref<Object> get_parts                  (PyDate* self, void*);
  static ref<Object> get_valid                  (PyDate* self, void*);
  static ref<Object> get_week                   (PyDate* self, void*);
  static ref<Object> get_week_year              (PyDate* self, void*);
  static ref<Object> get_weekday                (PyDate* self, void*);
  static ref<Object> get_year                   (PyDate* self, void*);
  static ref<Object> get_ymdi                   (PyDate* self, void*);
  static GetSets<PyDate> tp_getsets_;

  /** Date format used to generate the repr.  */
  static unique_ptr<cron::DateFormat> repr_format_;

  static Type build_type(string const& type_name);

public:

  static Type type_;

};


template<typename DATE>
void
PyDate<DATE>::add_to(
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
  Dict* const dict = (Dict*) type_.tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("INVALID" , create(Date::INVALID));
  dict->SetItemString("MAX"     , create(Date::MAX));
  dict->SetItemString("MIN"     , create(Date::MIN));
  dict->SetItemString("MISSING" , create(Date::MISSING));

  // Add the type to the module.
  module.add(&type_);
}


template<typename DATE>
ref<PyDate<DATE>>
PyDate<DATE>::create(
  Date date,
  PyTypeObject* type)
{
  auto obj = ref<PyDate>::take(check_not_null(PyDate::type_.tp_alloc(type, 0)));

  // date_ is const to indicate immutablity, but Python initialization is later
  // than C++ initialization, so we have to cast off const here.
  new(const_cast<Date*>(&obj->date_)) Date{date};
  return obj;
}


template<typename DATE>
bool
PyDate<DATE>::Check(
  PyObject* other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

template<typename DATE>
void
PyDate<DATE>::tp_init(
  PyDate* const self, 
  Tuple* const args, 
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("function takes no keyword arguments");
  auto const num_args = args->Length();
  Date date;
  if (num_args == 0)
    ;
  else if (num_args == 1)
    date = convert_to_date<Date>(args->GetItem(0));
  else if (num_args == 2)
    date = ordinal_parts_to_date<Date>(args);
  else if (num_args == 3)
    date = parts_to_date<Date>(args);
  else
    throw TypeError("function takes 0, 1, 2, or 3 arguments");

  new(self) PyDate{date};
}


template<typename DATE>
void
PyDate<DATE>::tp_dealloc(
  PyDate* const self)
{
  self->date_.~DateTemplate();
  self->ob_type->tp_free(self);
}


template<typename DATE>
ref<Unicode>
PyDate<DATE>::tp_repr(
  PyDate* const self)
{
  return Unicode::from((*repr_format_)(self->date_));
}


template<typename DATE>
ref<Unicode>
PyDate<DATE>::tp_str(
  PyDate* const self)
{
  // FIXME: Make the format configurable.
  auto& format = cron::DateFormat::get_default();
  return Unicode::from(format(self->date_));
}


template<typename DATE>
ref<Object>
PyDate<DATE>::tp_richcompare(
  PyDate* const self,
  Object* const other,
  int const comparison)
{
  auto const other_date = maybe_date<Date>(other);
  if (!other_date)
    return not_implemented_ref();

  Date const d0 = self->date_;
  Date const d1 = *other_date;

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
  return Bool::from(result);
}


//------------------------------------------------------------------------------
// Number methods
//------------------------------------------------------------------------------

// FIXME: Should (MISSING - 1) -> INVALID?

template<typename DATE>
inline ref<Object>
PyDate<DATE>::nb_add(
  PyDate* const self,
  Object* const other,
  bool /* ignored */)
{
  auto offset = other->maybe_long_value();
  if (offset)
    return 
      *offset == 0 ? ref<PyDate>::of(self)
      : create(self->date_ + *offset, self->ob_type);
  else
    return not_implemented_ref();
}


template<typename DATE>
inline ref<Object>
PyDate<DATE>::nb_subtract(
  PyDate* const self,
  Object* const other,
  bool right)
{
  if (right) 
    return not_implemented_ref();

  auto const other_date = maybe_date<Date>(other);
  if (other_date)
    if (self->date_.is_valid() && other_date->is_valid())
      return Long::FromLong(self->date_ - *other_date);
    else
      return none_ref();

  auto offset = other->maybe_long_value();
  if (offset)
    return 
      *offset == 0 
      ? ref<PyDate>::of(self)  // Optimization: same date.
      : create(self->date_ - *offset, self->ob_type);

  return not_implemented_ref();
}


template<typename DATE>
PyNumberMethods
PyDate<DATE>::tp_as_number_ = {
  (binaryfunc)  wrap<PyDate, nb_add>,           // nb_add
  (binaryfunc)  wrap<PyDate, nb_subtract>,      // nb_subtract
  (binaryfunc)  nullptr,                        // nb_multiply
  (binaryfunc)  nullptr,                        // nb_remainder
  (binaryfunc)  nullptr,                        // nb_divmod
  (ternaryfunc) nullptr,                        // nb_power
  (unaryfunc)   nullptr,                        // nb_negative
  (unaryfunc)   nullptr,                        // nb_positive
  (unaryfunc)   nullptr,                        // nb_absolute
  (inquiry)     nullptr,                        // nb_bool
  (unaryfunc)   nullptr,                        // nb_invert
  (binaryfunc)  nullptr,                        // nb_lshift
  (binaryfunc)  nullptr,                        // nb_rshift
  (binaryfunc)  nullptr,                        // nb_and
  (binaryfunc)  nullptr,                        // nb_xor
  (binaryfunc)  nullptr,                        // nb_or
  (unaryfunc)   nullptr,                        // nb_int
  (void*)       nullptr,                        // nb_reserved
  (unaryfunc)   nullptr,                        // nb_float
  (binaryfunc)  nullptr,                        // nb_inplace_add
  (binaryfunc)  nullptr,                        // nb_inplace_subtract
  (binaryfunc)  nullptr,                        // nb_inplace_multiply
  (binaryfunc)  nullptr,                        // nb_inplace_remainder
  (ternaryfunc) nullptr,                        // nb_inplace_power
  (binaryfunc)  nullptr,                        // nb_inplace_lshift
  (binaryfunc)  nullptr,                        // nb_inplace_rshift
  (binaryfunc)  nullptr,                        // nb_inplace_and
  (binaryfunc)  nullptr,                        // nb_inplace_xor
  (binaryfunc)  nullptr,                        // nb_inplace_or
  (binaryfunc)  nullptr,                        // nb_floor_divide
  (binaryfunc)  nullptr,                        // nb_true_divide
  (binaryfunc)  nullptr,                        // nb_inplace_floor_divide
  (binaryfunc)  nullptr,                        // nb_inplace_true_divide
  (unaryfunc)   nullptr,                        // nb_index
/* FIXME: Python 2.5
  (binaryfunc)  nullptr,                        // nb_matrix_multiply
  (binaryfunc)  nullptr,                        // nb_inplace_matrix_multiply
*/
};


//------------------------------------------------------------------------------
// Methods
//------------------------------------------------------------------------------

template<typename DATE>
ref<Object>
PyDate<DATE>::method_from_datenum(
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


template<typename DATE>
ref<Object>
PyDate<DATE>::method_from_ordinal_date(
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

  return create(Date::from_ordinal_date(year, ordinal - 1), type);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::method_from_parts(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("from_parts() takes no keyword arguments");

  auto const num_args = args->Length();
  Sequence* parts;
  // Accept either a single three-element sequence, or three args.
  if (num_args == 1) {
    parts = cast<Sequence>(args->GetItem(0));
    if (parts->Length() < 3)
      throw TypeError("parts must be a 3-element (or longer) sequence");
  }
  else if (num_args == 3)
    parts = args;
  else
    throw TypeError("from_parts() takes one or three arguments");

  return create(parts_to_date<Date>(parts), type);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::method_from_week_date(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] 
    = {"week_year", "week", "weekday", nullptr};
  cron::Year week_year;
  cron::Week week;
  cron::Weekday weekday;
  static_assert(sizeof(cron::Year) == sizeof(short), "year is not a short");
  static_assert(sizeof(cron::Week) == sizeof(char), "week is not a char");
  static_assert(sizeof(cron::Weekday) == sizeof(char), "week is not a char");
  Arg::ParseTupleAndKeywords(
    args, kw_args, "Hbb", arg_names, &week_year, &week, &weekday);

  return create(Date::from_week_date(week_year, week - 1, weekday), type);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::method_from_ymdi(
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
template<typename DATE>
ref<Object>
PyDate<DATE>::method_is_same(
  PyDate* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"other", nullptr};
  Object* other;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &other);

  auto const other_date = maybe_date<Date>(other);
  return Bool::from(other_date && self->date_.is(*other_date));
}


template<typename DATE>
Methods<PyDate<DATE>>
PyDate<DATE>::tp_methods_
  = Methods<PyDate>()
    .template add_class<method_from_datenum>        ("from_datenum")
    .template add_class<method_from_ordinal_date>   ("from_ordinal_date")
    .template add_class<method_from_parts>          ("from_parts")
    .template add_class<method_from_week_date>      ("from_week_date")
    .template add_class<method_from_ymdi>           ("from_ymdi")
    .template add<method_is_same>                   ("is_same")
  ;


//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

template<typename DATE>
ref<Object>
PyDate<DATE>::get_datenum(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_datenum());
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_day(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().day + 1);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_invalid(
  PyDate* const self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_invalid());
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_missing(
  PyDate* const self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_missing());
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_month(
  PyDate* const self,
  void* /* closure */)
{
  return get_month_obj(self->date_.get_parts().month + 1);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_ordinal(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().ordinal + 1);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_parts(
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
  parts_obj->initialize(5, Long::FromLong(parts.week + 1));
  parts_obj->initialize(6, get_weekday_obj(parts.weekday));
  return std::move(parts_obj);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_valid(
  PyDate* const self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_valid());
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_week(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().week);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_week_year(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().week_year);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_weekday(
  PyDate* const self,
  void* /* closure */)
{
  return get_weekday_obj(self->date_.get_parts().weekday);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_year(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_parts().year);
}


template<typename DATE>
ref<Object>
PyDate<DATE>::get_ymdi(
  PyDate* const self,
  void* /* closure */)
{
  auto const parts = self->date_.get_parts();
  int ymd = 10000 * parts.year + 100 * (parts.month + 1) + (parts.day + 1);
  return Long::FromLong(ymd);
}


template<typename DATE>
GetSets<PyDate<DATE>>
PyDate<DATE>::tp_getsets_ 
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
// Other members
//------------------------------------------------------------------------------

template<typename DATE>
unique_ptr<cron::DateFormat>
PyDate<DATE>::repr_format_;


//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

namespace {

/**
 * Assuming 'obj' is a PyDate<DATE>, returns its datenum.
 *
 * Used for the tp_print hack in 'build_type()' below.
 */
template<typename DATE>
cron::Datenum
_get_datenum(
  PyObject* obj)
{
  return static_cast<PyDate<DATE>*>(obj)->date_.get_datenum();
}


}  // anonymous namespace


template<typename DATE>
Type
PyDate<DATE>::build_type(
  string const& type_name)
{
  return PyTypeObject{
    PyVarObject_HEAD_INIT(nullptr, 0)
    (char const*)         strdup(type_name.c_str()),      // tp_name
    (Py_ssize_t)          sizeof(PyDate),                 // tp_basicsize
    (Py_ssize_t)          0,                              // tp_itemsize
    (destructor)          wrap<PyDate, tp_dealloc>,       // tp_dealloc
    // FIXME: Hack!  We'd like to provide a way for any PyDate instance to
    // return its datenum, for efficient manipulation by other PyDate instances,
    // without virtual methods.  PyTypeObject doesn't provide any slot for us to
    // stash this, so we requisition the deprecated tp_print slot.  This may
    // break in future Python versions, if that slot is reused.
    (printfunc)           &_get_datenum<DATE>,            // tp_print
    (getattrfunc)         nullptr,                        // tp_getattr
    (setattrfunc)         nullptr,                        // tp_setattr
                          nullptr,                        // tp_reserved
    (reprfunc)            wrap<PyDate, tp_repr>,          // tp_repr
    (PyNumberMethods*)    &tp_as_number_,                 // tp_as_number
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
    (richcmpfunc)         wrap<PyDate, tp_richcompare>,   // tp_richcompare
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


template<typename DATE>
Type
PyDate<DATE>::type_;


//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

using PyDateDefault = PyDate<cron::Date>;

inline ref<Object>
make_date(
  cron::Datenum const datenum,
  Object* type=(Object*) &PyDateDefault::type_)
{
  // Special case fast path for the default date type.
  if (type == (Object*) &PyDateDefault::type_)
    return PyDateDefault::create(PyDateDefault::Date::from_datenum(datenum));
  else 
    return type->CallMethodObjArgs("from_datenum", Long::FromLong(datenum));
}


template<typename DATE>
inline DATE
parts_to_date(
  Sequence* const parts)
{
  long const year   = parts->GetItem(0)->long_value();
  long const month  = parts->GetItem(1)->long_value();
  long const day    = parts->GetItem(2)->long_value();
  return DATE::from_parts(year, month - 1, day - 1);
}


template<typename DATE>
inline DATE
ordinal_parts_to_date(
  Sequence* const parts)
{
  long const year       = parts->GetItem(0)->long_value();
  long const ordinal    = parts->GetItem(1)->long_value();
  return DATE::from_ordinal_date(year, ordinal - 1);
}


template<typename DATE>
inline optional<DATE>
maybe_date(
  Object* const obj)
{
  if (PyDate<DATE>::Check(obj)) 
    // Exact wrapped type.
    return static_cast<PyDate<DATE>*>(obj)->date_;

  if (obj->ob_type->tp_print != nullptr) {
    // Each PyDate instantiation places the pointer to a function that returns
    // its datenum into the tp_print slot; see 'build_type()'.
    auto const get_datenum 
      = reinterpret_cast<cron::Datenum (*)(Object*)>(obj->ob_type->tp_print);
    return DATE::from_datenum(get_datenum(obj));
  }

  if (PyDate_Check(obj)) 
    return DATE::from_parts(
      PyDateTime_GET_YEAR(obj),
      PyDateTime_GET_MONTH(obj) - 1,
      PyDateTime_GET_DAY(obj) - 1);

  // Try for a date type that has a 'datenum' attribute.
  auto datenum = obj->GetAttrString("datenum", false);
  if (datenum != nullptr) 
    return DATE::from_datenum(datenum->long_value());

  // Try for a date type that as a 'toordinal()' method.
  auto ordinal = obj->CallMethodObjArgs("toordinal", false);
  if (ordinal != nullptr)
    return DATE::from_datenum(ordinal->long_value() - 1);

  // No type match.
  return {};
}


template<typename DATE>
inline DATE
convert_to_date(
  Object* const obj)
{
  if (obj == None) 
    // Use the default value.
    return DATE{};

  auto date = maybe_date<DATE>(obj);
  if (date)
    return *date;

  if (Sequence::Check(obj)) {
    auto seq = static_cast<Sequence*>(obj);
    if (seq->Length() == 3) 
      // Interpret a three-element sequence as date parts.
      return parts_to_date<DATE>(seq);
    else if (seq->Length() == 2) 
      // Interpret a two-element sequence as ordinal parts.
      return ordinal_parts_to_date<DATE>(seq);
  }

  auto const long_obj = obj->Long(false);
  if (long_obj != nullptr) {
    // Interpret eight-digit values as YMDI.
    long const ymdi = (long) *long_obj;
    if (10000000 <= ymdi && ymdi <= 99999999) 
      return DATE::from_ymdi(ymdi);
  }

  // FIXME: Parse strings.

  throw py::TypeError("can't convert to a date"s + *obj->Repr());
}


//------------------------------------------------------------------------------

}  // namespace alxs


