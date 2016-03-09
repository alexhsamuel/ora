#pragma once

#include <cmath>
#include <experimental/optional>
#include <iostream>

#include <datetime.h>

#include "cron/format.hh"
#include "cron/math.hh"
#include "cron/time.hh"
#include "cron/time_zone.hh"
#include "py.hh"
#include "PyDate.hh"
#include "PyDaytime.hh"
#include "PyTime.hh"
#include "PyTimeZone.hh"

namespace alxs {

using namespace py;

using std::experimental::optional;
using std::make_unique;
using std::string;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

StructSequenceType* get_time_parts_type();

// template<typename TIME> optional<TIME> convert_object(Object*);
template<typename TIME> TIME convert_time_object(Object*);

//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

template<typename TIME>
class PyTime
  : public ExtensionType
{
public:

  using Time = TIME;

  /** 
   * Readies the Python type and adds it to `module` as `name`.  
   *
   * Should only be called once; this is not checked.
   */
  static void add_to(Module& module, string const& name);

  static Type type_;

  /**
   * Creates an instance of the Python type.
   */
  static ref<PyTime> create(Time time, PyTypeObject* type=&type_);

  /**
   * Returns true if 'object' is an instance of this type.
   */
  static bool Check(PyObject* object);

  PyTime(Time time) : time_(time) {}

  /**
   * The wrapped date instance.
   *
   * This is the only non-static data member.
   */
  Time const time_;

private:

  static void tp_init(PyTime*, Tuple* args, Dict* kw_args);
  static void tp_dealloc(PyTime*);
  static ref<Unicode> tp_repr(PyTime*);
  static ref<Unicode> tp_str(PyTime*);
  static ref<Object> tp_richcompare(PyTime*, Object*, int);

  // Number methods.
  static ref<Object> nb_matrix_multiply         (PyTime*, Object*, bool);
  static PyNumberMethods tp_as_number_;

  // Methods.
  static ref<Object> method__from_local             (PyTypeObject*, Tuple*, Dict*);
  static ref<Object> method__to_local               (PyTime*,       Tuple*, Dict*);
  static ref<Object> method_get_date_daytime        (PyTime*,       Tuple*, Dict*);
  static ref<Object> method_get_datenum_daytick     (PyTime*,       Tuple*, Dict*);
  static ref<Object> method_get_parts               (PyTime*,       Tuple*, Dict*);
  static ref<Object> method_is_same                 (PyTime*,       Tuple*, Dict*);
  static Methods<PyTime> tp_methods_;

  // Getsets.
  static ref<Object> get_invalid                    (PyTime*, void*);
  static ref<Object> get_missing                    (PyTime*, void*);
  static ref<Object> get_offset                     (PyTime*, void*);
  static ref<Object> get_timetick                   (PyTime*, void*);
  static ref<Object> get_valid                      (PyTime*, void*);
  static GetSets<PyTime> tp_getsets_;

  /** Date format used to generate the repr.  */
  static unique_ptr<cron::TimeFormat> repr_format_;
  /** Date format used to generate the str.  */
  static unique_ptr<cron::TimeFormat> str_format_;

  static Type build_type(string const& type_name);

};


template<typename TIME>
void
PyTime<TIME>::add_to(
  Module& module,
  string const& name)
{
  // Construct the type struct.
  type_ = build_type(string{module.GetName()} + "." + name);
  // Hand it to Python.
  type_.Ready();

  // Build the repr format.
  repr_format_ = make_unique<cron::TimeFormat>(
    name + "(%0Y, %0m, %0d, %H, %M, %S)",  // FIXME: Not a ctor.
    name + ".INVALID",
    name + ".MISSING");

  // Build the str format.  Choose precision for seconds that captures actual
  // precision of the time class.
  std::string pattern = "%Y-%m-%dT%H:%M:%";
  size_t const precision = (size_t) log10(Time::DENOMINATOR);
  if (precision > 0) {
    pattern += ".";
    pattern += std::to_string(precision);
  }
  pattern += "SZ";
  str_format_ = make_unique<cron::TimeFormat>(pattern);

  // Add in static data members.
  Dict* const dict = (Dict*) type_.tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("INVALID" , create(Time::INVALID));
  dict->SetItemString("MAX"     , create(Time::MAX));
  dict->SetItemString("MIN"     , create(Time::MIN));
  dict->SetItemString("MISSING" , create(Time::MISSING));

  // Add the type to the module.
  module.add(&type_);
}


template<typename TIME>
ref<PyTime<TIME>>
PyTime<TIME>::create(
  Time const time,
  PyTypeObject* const type)
{
  auto obj = ref<PyTime>::take(check_not_null(PyTime::type_.tp_alloc(type, 0)));

  // time_ is const to indicate immutablity, but Python initialization is later
  // than C++ initialization, so we have to cast off const here.
  new(const_cast<Time*>(&obj->time_)) Time{time};
  return obj;
}


template<typename TIME>
Type
PyTime<TIME>::type_;


template<typename TIME>
bool
PyTime<TIME>::Check(
  PyObject* const other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

template<typename TIME>
void
PyTime<TIME>::tp_init(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  // FIXME
  typename Time::Offset offset;
  Arg::ParseTuple(args, "|k", &offset);

  new(self) PyTime(Time::from_offset(offset));
}


template<typename TIME>
void
PyTime<TIME>::tp_dealloc(
  PyTime* const self)
{
  self->time_.~TimeTemplate();
  self->ob_type->tp_free(self);
}


template<typename TIME>
ref<Unicode>
PyTime<TIME>::tp_repr(
  PyTime* const self)
{
  return Unicode::from((*repr_format_)(self->time_, cron::UTC));
}


template<typename TIME>
ref<Unicode>
PyTime<TIME>::tp_str(
  PyTime* const self)
{
  // FIXME: Not UTC?
  return Unicode::from((*str_format_)(self->time_, cron::UTC));  
}


template<typename TIME>
ref<Object>
PyTime<TIME>::tp_richcompare(
  PyTime* const self,
  Object* const other,
  int const comparison)
{
  Time const t0 = self->time_;
  Time t1;
  try {
    t1 = convert_time_object<Time>(other);
  } catch (Exception) {
    Exception::Clear();
    return not_implemented_ref();
  }

  bool result;
  switch (comparison) {
  case Py_EQ: result = t0 == t1; break;
  case Py_GE: result = t0 >= t1; break;
  case Py_GT: result = t0 >  t1; break;
  case Py_LE: result = t0 <= t1; break;
  case Py_LT: result = t0 <  t1; break;
  case Py_NE: result = t0 != t1; break;
  default:    result = false; assert(false);
  }
  return Bool::from(result);
}


//------------------------------------------------------------------------------
// Number methods
//------------------------------------------------------------------------------

namespace {

// FIXME: Use a LocalTime object instead of a pair.
inline ref<Object>
make_date_daytime(
  cron::Datenum const datenum,
  cron::Daytick const daytick)
{
  auto result = Tuple::New(2);
  result->initialize(
    0, PyDate<cron::Date>::create(cron::Date::from_datenum(datenum)));
  result->initialize(
    1, PyDaytime<cron::Daytime>::create(cron::Daytime::from_daytick(daytick)));
  return std::move(result);
}


}  // anonymous namespace

template<typename TIME>
inline ref<Object>
PyTime<TIME>::nb_matrix_multiply(
  PyTime* const self,
  Object* const other,
  bool right)
{
  if (right || !PyTimeZone::Check(other))
    return not_implemented_ref();
  else {
    auto tz = *cast<PyTimeZone>(other)->tz_;
    auto local = cron::to_local_datenum_daytick(self->time_, tz);
    return make_date_daytime(local.datenum, local.daytick);
  }
}


template<typename TIME>
PyNumberMethods
PyTime<TIME>::tp_as_number_ = {
  (binaryfunc)  nullptr,                        // nb_add
  (binaryfunc)  nullptr,                        // nb_subtract
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
#if PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 5
  (binaryfunc)  wrap<PyTime, nb_matrix_multiply>, // nb_matrix_multiply
  (binaryfunc)  nullptr,                        // nb_inplace_matrix_multiply
#endif
};


//------------------------------------------------------------------------------
// Methods
//------------------------------------------------------------------------------

template<typename TIME>
ref<Object>
PyTime<TIME>::method__from_local(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] 
    = {"datenum", "daytick", "time_zone", "first", nullptr};
  cron::Datenum datenum;
  cron::Daytick daytick;
  Object* tz_arg;
  int first = true;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "IkO|p", arg_names, &datenum, &daytick, &tz_arg, &first);

  auto tz = to_time_zone(tz_arg);
  auto t = cron::from_local<TIME>(datenum, daytick, tz, first);
  return create(t, type);
}


template<typename TIME>
ref<Object>
PyTime<TIME>::method__to_local(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"time_zone", nullptr};
  Object* tz;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &tz);

  auto local = cron::to_local_datenum_daytick(self->time_, to_time_zone(tz));
  ref<Tuple> result = Tuple::New(2);
  result->initialize(0, Long::FromLong(local.datenum));
  result->initialize(1, Long::FromUnsignedLong(local.daytick));
  return std::move(result);
}


template<typename TIME>
ref<Object>
PyTime<TIME>::method_get_date_daytime(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  // FIXME: Pass in the Date and Daytime class to use as keyword arguments.
  static char const* const arg_names[] = {"tz", nullptr};
  Object* tz_arg;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "O", arg_names, &tz_arg);

  auto tz = to_time_zone(tz_arg);
  auto local = cron::to_local_datenum_daytick(self->time_, tz);
  return make_date_daytime(local.datenum, local.daytick);
}


template<typename TIME>
ref<Object>
PyTime<TIME>::method_get_datenum_daytick(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"tz", nullptr};
  Object* tz_arg;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &tz_arg);

  auto tz = to_time_zone(tz_arg);
  auto local = cron::to_local_datenum_daytick(self->time_, tz);

  auto result = Tuple::New(2);
  result->initialize(0, Long::FromLong(local.datenum));
  result->initialize(1, Long::FromLong(local.daytick));
  return std::move(result);
}


template<typename TIME>
ref<Object>
PyTime<TIME>::method_get_parts(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"tz", nullptr};
  Object* tz_arg;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &tz_arg);

  auto tz = to_time_zone(tz_arg);
  auto parts = self->time_.get_parts(tz);

  auto date_parts = get_date_parts_type()->New();
  date_parts->initialize(0, Long::FromLong(parts.date.year));
  date_parts->initialize(1, get_month_obj(parts.date.month + 1));
  date_parts->initialize(2, Long::FromLong(parts.date.day + 1));
  date_parts->initialize(3, Long::FromLong(parts.date.ordinal + 1));
  date_parts->initialize(4, Long::FromLong(parts.date.week_year));
  date_parts->initialize(5, Long::FromLong(parts.date.week + 1));
  date_parts->initialize(6, get_weekday_obj(parts.date.weekday));

  auto daytime_parts = get_daytime_parts_type()->New();
  daytime_parts->initialize(0, Long::FromLong(parts.daytime.hour));
  daytime_parts->initialize(1, Long::FromLong(parts.daytime.minute));
  daytime_parts->initialize(2, Float::FromDouble(parts.daytime.second));

  auto time_zone_parts = get_time_zone_parts_type()->New();
  time_zone_parts->initialize(0, Long::FromLong(parts.time_zone.offset));
  time_zone_parts->initialize(1, Unicode::from(parts.time_zone.abbreviation));
  time_zone_parts->initialize(2, Bool::from(parts.time_zone.is_dst));

  auto time_parts = get_time_parts_type()->New();
  time_parts->initialize(0, std::move(date_parts));
  time_parts->initialize(1, std::move(daytime_parts));
  time_parts->initialize(2, std::move(time_zone_parts));

  return std::move(time_parts);
}


// We call this method "is_same" because "is" is a keyword in Python.
template<typename TIME>
ref<Object>
PyTime<TIME>::method_is_same(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"object", nullptr};
  Object* object;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &object);

  Time time;
  try {
    time = convert_time_object<Time>(object);
  }
  catch (Exception) {
    Exception::Clear();
    return Bool::from(false);
  }
  return Bool::from(self->time_.is(time));
}


template<typename TIME>
Methods<PyTime<TIME>>
PyTime<TIME>::tp_methods_
  = Methods<PyTime>()
    .template add_class<method__from_local>             ("_from_local")
    .template add<method__to_local>                     ("_to_local")
    .template add<method_get_date_daytime>              ("get_date_daytime")
    .template add<method_get_datenum_daytick>           ("get_datenum_daytick")
    .template add<method_get_parts>                     ("get_parts")
    .template add<method_is_same>                       ("is_same")
  ;


//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

template<typename TIME>
ref<Object>
PyTime<TIME>::get_invalid(
  PyTime* const self,
  void* /* closure */)
{
  return Bool::from(self->time_.is_invalid());
}


template<typename TIME>
ref<Object>
PyTime<TIME>::get_missing(
  PyTime* const self,
  void* /* closure */)
{
  return Bool::from(self->time_.is_missing());
}


template<typename TIME>
ref<Object>
PyTime<TIME>::get_offset(
  PyTime* const self,
  void* /* closure */)
{
  return Long::FromUnsignedLong(self->time_.get_offset());
}


template<typename TIME>
ref<Object>
PyTime<TIME>::get_timetick(
  PyTime* const self,
  void* /* closure */)
{
  return Long::from(self->time_.get_timetick());
}


template<typename TIME>
ref<Object>
PyTime<TIME>::get_valid(
  PyTime* const self,
  void* /* closure */)
{
  return Bool::from(self->time_.is_valid());
}


template<typename TIME>
GetSets<PyTime<TIME>>
PyTime<TIME>::tp_getsets_ 
  = GetSets<PyTime>()
    .template add_get<get_invalid>      ("invalid")
    .template add_get<get_missing>      ("missing")
    .template add_get<get_offset>       ("offset")
    .template add_get<get_timetick>     ("timetick")
    .template add_get<get_valid>        ("valid")
  ;


//------------------------------------------------------------------------------
// Other members
//------------------------------------------------------------------------------

template<typename TIME>
unique_ptr<cron::TimeFormat>
PyTime<TIME>::repr_format_;


template<typename TIME>
unique_ptr<cron::TimeFormat>
PyTime<TIME>::str_format_;


//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

template<typename TIME>
Type
PyTime<TIME>::build_type(
  string const& type_name)
{
  return PyTypeObject{
    PyVarObject_HEAD_INIT(nullptr, 0)
    (char const*)         strdup(type_name.c_str()),      // tp_name
    (Py_ssize_t)          sizeof(PyTime),                 // tp_basicsize
    (Py_ssize_t)          0,                              // tp_itemsize
    (destructor)          wrap<PyTime, tp_dealloc>,       // tp_dealloc
    (printfunc)           nullptr,                        // tp_print
    (getattrfunc)         nullptr,                        // tp_getattr
    (setattrfunc)         nullptr,                        // tp_setattr
                          nullptr,                        // tp_reserved
    (reprfunc)            wrap<PyTime, tp_repr>,          // tp_repr
    (PyNumberMethods*)    &tp_as_number_,                 // tp_as_number
    (PySequenceMethods*)  nullptr,                        // tp_as_sequence
    (PyMappingMethods*)   nullptr,                        // tp_as_mapping
    (hashfunc)            nullptr,                        // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            wrap<PyTime, tp_str>,           // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         nullptr,                        // tp_doc
    (traverseproc)        nullptr,                        // tp_traverse
    (inquiry)             nullptr,                        // tp_clear
    (richcmpfunc)         wrap<PyTime, tp_richcompare>,   // tp_richcompare
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
    (initproc)            wrap<PyTime, tp_init>,          // tp_init
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


//------------------------------------------------------------------------------
// Helpers

using PyTimeDefault = PyTime<cron::Time>;

/**
 * From Python time and time zone objects, returns the local datenum and 
 * daytick.
 */
inline cron::LocalDatenumDaytick
to_local(
  Object* const time,
  Object* const time_zone)
{
  if (PyTime<cron::Time>::Check(time)) 
    // Special case fast path for the default time type.
    return cron::to_local_datenum_daytick(
      cast<PyTime<cron::Time>>(time)->time_, to_time_zone(time_zone));

  else {
    // Call its _to_local() method.
    auto local 
      = cast<Sequence>(time->CallMethodObjArgs("_to_local", time_zone));
    return {
      (cron::Datenum) (long) *cast<Long>(local->GetItem(0)),
      (cron::Daytick) (long) *cast<Long>(local->GetItem(1)),
    };
  }
}


/**
 * Attempts to convert various kinds of Python time objects to Time.
 *
 * If 'obj' is a time object, returns the equivalent time.  Otherwise, throws
 * 'Exception'.
 */
template<typename TIME>
TIME
convert_time_object(
  Object* const obj)
{
  if (obj == nullptr)
    // Use the default value.
    return TIME();

  if (PyTime<TIME>::Check(obj))
    // Exact wrapped type.
    return cast<PyTime<TIME>>(obj)->time_;

  // FIXME: Check for other PyTime types.
  // When doing so, interpret INVALID and MISSING.

  // If the type provides timetick, use that.
  auto timetick_obj = obj->GetAttrString("timetick", false);
  if (timetick_obj != nullptr) 
    return TIME::from_timetick((cron::Timetick) *timetick_obj->Long());

  // Try to handle it like a `datetime.datetime`.
  if (PyDateTime_Check(obj)) {
    // First, make sure it's localized.
    auto const tzinfo = obj->GetAttrString("tzinfo", false);
    if (tzinfo == Py_None)
      throw py::ValueError("unlocalized datetime doesn't represent a time");
    auto const tz_name = tzinfo->GetAttrString("zone")->Str()->as_utf8_string();
    auto const tz = cron::get_time_zone(tz_name);
    
    // FIXME: Provide a all-integer ctor with (sec, usec).
    return TIME(
      PyDateTime_GET_YEAR(obj),
      PyDateTime_GET_MONTH(obj) - 1,
      PyDateTime_GET_DAY(obj) - 1,
      PyDateTime_DATE_GET_HOUR(obj),
      PyDateTime_DATE_GET_MINUTE(obj),
      PyDateTime_DATE_GET_SECOND(obj) 
      + PyDateTime_DATE_GET_MICROSECOND(obj) * 1e-6,
      tz,
      true);
  }

  throw py::TypeError("not a time");
}


//------------------------------------------------------------------------------

}  // namespace alxs

