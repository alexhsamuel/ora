#pragma once

#include <cmath>
#include <iostream>
#include <unordered_map>

#include <Python.h>
#include <datetime.h>

#include "ora/lib/math.hh"
#include "ora.hh"
#include "py.hh"
#include "PyDate.hh"
#include "PyDaytime.hh"
#include "PyTime.hh"
#include "PyTimeZone.hh"
#include "util.hh"

namespace ora {
namespace py {

using namespace std::literals;

using std::make_unique;
using std::string;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

StructSequenceType* get_time_parts_type();

/**
 * Helper for converting a (localtime, tz) sequence to a time.
 */
template<class TIME> inline TIME localtime_to_time(Sequence*);

/**
 * Helper for converting a (date, daytime, tz) sequence to a time.
 */
template<class TIME> inline TIME date_daytime_to_time(Sequence*);

/**
 * Helper for converting a (Y,m,d,H,M,S,tz) sequence to a time.
 */
template<class TIME> inline TIME parts_to_time(Sequence*);

/**
 * Attempts to decode various time objects.  The following objects are
 * recognized:
 *
 *   - PyTime instances
 *   - datetime.datetime instances
 */
template<class TIME> inline std::pair<bool, TIME> maybe_time(Object*);

/**
 * Converts an object to a time.  Beyond 'maybe_time()', recognizes the 
 * following: 
 *
 *   - 
 *
 * If the argument cannot be converted, raises a Python exception.
 */
template<class TIME> inline TIME convert_to_time(Object*);

//------------------------------------------------------------------------------
// Virtual API
//------------------------------------------------------------------------------

/*
 * Provides an API with dynamic dispatch to PyTime objects.
 * 
 * The PyTime class, since it is a Python type, cannot be a C++ virtual class.
 * This mechanism interfaces the C++ virtual method mechanism with the Python
 * type system by mapping the Python type to a stub virtual C++ class.
 */
class PyTimeAPI
{
public:

  /*
   * Registers a virtual API for a Python type.
   */
  static void add(PyTypeObject* const type, std::unique_ptr<PyTimeAPI>&& api) 
    { apis_.emplace(type, std::move(api)); }

  /*
   * Returns the API for a Python object, or nullptr if it isn't a PyTime.
   */
  static PyTimeAPI const*
  get(
    PyTypeObject* const type)
  {
    auto api = apis_.find(type);
    return api == apis_.end() ? nullptr : api->second.get();
  }

  static PyTimeAPI const* get(PyObject* const obj)
    { return get(obj->ob_type);  }

  // API methods.
  virtual ref<Object>               from_local_datenum_daytick(ora::Datenum, ora::Daytick, ora::TimeZone const&, bool) const = 0; 
  virtual int64_t                   get_epoch_time(Object* time) const = 0;
  virtual ora::time::Time128       get_time128(Object* time) const = 0;
  virtual bool                      is_invalid(Object* time) const = 0;
  virtual bool                      is_missing(Object* time) const = 0;
  virtual ref<Object>               now() const = 0;
  virtual ora::LocalDatenumDaytick to_local_datenum_daytick(Object* time, ora::TimeZone const& tz) const = 0;

private:

  static std::unordered_map<PyTypeObject*, std::unique_ptr<PyTimeAPI>> apis_;

};


//------------------------------------------------------------------------------
// Docstrings
//------------------------------------------------------------------------------

namespace docstring {

using doct_t = char const* const;

namespace pytime {

#include "PyTime.docstrings.hh.inc"

}  // namespace docstring

}  // namespace pytime


//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

template<class TIME>
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

  class API 
  : public PyTimeAPI 
  {
  public:

    virtual int64_t get_epoch_time(Object* const time) const
      { return ora::time::get_epoch_time(((PyTime*) time)->time_); }

    virtual ora::time::Time128 get_time128(Object* const time) const
      { return ora::time::Time128(((PyTime*) time)->time_); }

    virtual ref<Object> from_local_datenum_daytick(ora::Datenum const datenum, ora::Daytick const daytick, ora::TimeZone const& tz, bool first) const
      { return PyTime::create(ora::from_local<Time>(datenum, daytick, tz, first)); }

    virtual ref<Object> now() const
      { return PyTime::create(ora::time::now<Time>()); }

    virtual bool is_invalid(Object* const time) const
      { return ((PyTime*) time)->time_.is_invalid(); }

    virtual bool is_missing(Object* const time) const
      { return ((PyTime*) time)->time_.is_missing(); }

    virtual ora::LocalDatenumDaytick to_local_datenum_daytick(Object* const time, ora::TimeZone const& tz) const
      { return ora::time::to_local_datenum_daytick(((PyTime*) time)->time_, tz); }

  };

private:

  static void           tp_init(PyTime*, Tuple* args, Dict* kw_args);
  static void           tp_dealloc(PyTime*);
  static ref<Unicode>   tp_repr(PyTime*);
  static Py_hash_t      tp_hash(PyTime*);
  static ref<Unicode>   tp_str(PyTime*);
  static ref<Object>    tp_richcompare(PyTime*, Object*, int);

  // Number methods.
  static ref<Object>    nb_add                  (PyTime*, Object*, bool);
  static ref<Object>    nb_subtract             (PyTime*, Object*, bool);
  static PyNumberMethods tp_as_number_;

  // Methods.
  static ref<Object>    method___format__       (PyTime*, Tuple*, Dict*);
  static ref<Object>    method_format           (PyTime*, Tuple*, Dict*);
  static ref<Object>    method_from_offset      (PyTypeObject*, Tuple*, Dict*);
  static ref<Object>    method_get_parts        (PyTime*, Tuple*, Dict*);
  static Methods<PyTime> tp_methods_;

  // Getsets.
  static ref<Object> get_invalid                (PyTime*, void*);
  static ref<Object> get_missing                (PyTime*, void*);
  static ref<Object> get_offset                 (PyTime*, void*);
  static ref<Object> get_std                    (PyTime*, void*);
  static ref<Object> get_valid                  (PyTime*, void*);
  static GetSets<PyTime> tp_getsets_;

  /** Date format used to generate the repr.  */
  static unique_ptr<ora::time::TimeFormat> repr_format_;
  /** Date format used to generate the str.  */
  static unique_ptr<ora::time::TimeFormat> str_format_;

  static Type build_type(string const& type_name);

};


//------------------------------------------------------------------------------

template<class TIME>
void
PyTime<TIME>::add_to(
  Module& module,
  string const& name)
{
  // Construct the type struct.
  type_ = build_type(string{module.GetName()} + "." + name);
  // Hand it to Python.
  type_.Ready();

  PyTimeAPI::add(&type_, std::make_unique<API>());

  // Choose precision for seconds that captures actual precision of the time
  // class (up to 1 fs).
  auto const precision 
    = std::min((size_t) ceil(log10((long double) Time::DENOMINATOR)), 15ul);

  // Build the repr format.
  repr_format_ = make_unique<ora::time::TimeFormat>(
    name + "(%0Y, %0m, %0d, %0H, %0M, %0." + std::to_string(precision) + "S, UTC)",
    name + ".INVALID",
    name + ".MISSING");

  // Build the str format.  
  std::string pattern = "%Y-%m-%dT%H:%M:%";
  if (precision > 0) {
    pattern += ".";
    pattern += std::to_string(precision);
  }
  pattern += "SZ";
  str_format_ = make_unique<ora::time::TimeFormat>(pattern);

  // Add in static data members.
  Dict* const dict = (Dict*) type_.tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("DENOMINATOR" , Long::from(Time::DENOMINATOR));
  dict->SetItemString("INVALID"     , create(Time::INVALID));
  dict->SetItemString("MAX"         , create(Time::MAX));
  dict->SetItemString("MIN"         , create(Time::MIN));
  dict->SetItemString("MISSING"     , create(Time::MISSING));
  dict->SetItemString("RESOLUTION"  , Float::FromDouble(1.0 / Time::DENOMINATOR));

  // Add the type to the module.
  module.add(&type_);
}


template<class TIME>
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


template<class TIME>
Type
PyTime<TIME>::type_;


template<class TIME>
bool
PyTime<TIME>::Check(
  PyObject* const other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

template<class TIME>
void
PyTime<TIME>::tp_init(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("function takes no keyword arguments");
  auto const num_args = args->Length();
  Time time;
  if (num_args == 0)
    ;
  else if (num_args == 1)
    time = convert_to_time<Time>(args->GetItem(0));
  else if (num_args == 2)
    time = localtime_to_time<Time>(args);
  else if (num_args == 3)
    time = date_daytime_to_time<Time>(args);
  else if (num_args == 7)
    time = parts_to_time<Time>(args);
  else
    throw TypeError("function takes 0, 1, 2, 3, or 7 arguments");

  new(self) PyTime(time);
}


template<class TIME>
void
PyTime<TIME>::tp_dealloc(
  PyTime* const self)
{
  self->time_.~TimeType();
  self->ob_type->tp_free(self);
}


template<class TIME>
ref<Unicode>
PyTime<TIME>::tp_repr(
  PyTime* const self)
{
  return Unicode::from((*repr_format_)(self->time_, *ora::UTC));
}


template<class TIME>
Py_hash_t
PyTime<TIME>::tp_hash(
  PyTime* const self)
{
  return 
      self->time_.is_invalid() ? std::numeric_limits<Py_hash_t>::max()
    : self->time_.is_missing() ? std::numeric_limits<Py_hash_t>::max() - 1
    : self->time_.get_offset();
}


template<class TIME>
ref<Unicode>
PyTime<TIME>::tp_str(
  PyTime* const self)
{
  // FIXME: Not UTC?
  return Unicode::from((*str_format_)(self->time_, *ora::UTC));  
}


template<class TIME>
ref<Object>
PyTime<TIME>::tp_richcompare(
  PyTime* const self,
  Object* const other,
  int const comparison)
{
  auto const other_time = maybe_time<Time>(other);
  if (!other_time.first)
    return not_implemented_ref();
  return richcmp(self->time_, other_time.second, comparison);
}


//------------------------------------------------------------------------------
// Number methods
//------------------------------------------------------------------------------

template<class TIME>
inline ref<Object>
PyTime<TIME>::nb_add(
  PyTime* const self,
  Object* const other,
  bool /* ignored */)
{
  auto offset = other->maybe_double_value();
  if (offset)
    return 
      *offset == 0 ? ref<PyTime>::of(self)
      : create(self->time_ + *offset, self->ob_type);
  else
    return not_implemented_ref();
}


template<class TIME>
inline ref<Object>
PyTime<TIME>::nb_subtract(
  PyTime* const self,
  Object* const other,
  bool right)
{
  if (right) 
    return not_implemented_ref();

  auto const other_time = maybe_time<Time>(other);
  if (other_time.first)
    if (self->time_.is_valid() && other_time.second.is_valid())
      return Float::from(self->time_ - other_time.second);
    else
      return none_ref();

  auto offset = other->maybe_double_value();
  if (offset)
    return 
      *offset == 0 
      ? ref<PyTime>::of(self)  // Optimization: same time.
      : create(self->time_ - *offset, self->ob_type);

  return not_implemented_ref();
}


template<class TIME>
PyNumberMethods
PyTime<TIME>::tp_as_number_ = {
  (binaryfunc)  wrap<PyTime, nb_add>,           // nb_add
  (binaryfunc)  wrap<PyTime, nb_subtract>,      // nb_subtract
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
  (binaryfunc)  nullptr,                        // nb_matrix_multiply
  (binaryfunc)  nullptr,                        // nb_inplace_matrix_multiply
#endif
};


//------------------------------------------------------------------------------
// Methods
//------------------------------------------------------------------------------

template<class TIME>
ref<Object>
PyTime<TIME>::method___format__(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  if (args->GetLength() != 1 || kw_args != nullptr)
    throw TypeError("__format__() takes one argument");
  auto const fmt = args->GetItem(0)->Str()->as_utf8();

  // If the format pattern is empty, use the default str format.
  if (*fmt == '\0')
    return Unicode::from((*str_format_)(self->time_));
  else
    return Unicode::from(ora::time::LocalTimeFormat::parse(fmt)(self->time_));
}


template<class TIME>
ref<Object>
PyTime<TIME>::method_format(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"pattern", "time_zone", nullptr};
  char* pattern;
  Object* time_zone = nullptr;
  Arg::ParseTupleAndKeywords(
    args, kw_args, "s|O", arg_names, &pattern, &time_zone);

  auto const tz = 
    time_zone == nullptr ? ora::UTC : convert_to_time_zone(time_zone);
  return Unicode::from(ora::time::TimeFormat(pattern)(self->time_, *tz));
}


template<class TIME>
ref<Object>
PyTime<TIME>::method_from_offset(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"from_offset", nullptr};
  Object* offset_arg;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &offset_arg);
  
  // int128_t still fits the valid range of Time128::Offset (i.e. uint128_t)
  // comfortably, due to the the overall date range limit.
  auto const offset = (int128_t) *offset_arg->Long();
  if (   offset < (int128_t) TIME::MIN.get_offset() 
      || offset > (int128_t) TIME::MAX.get_offset())
    throw OverflowError("time out of range");
  return create(TIME::from_offset((typename TIME::Offset) offset));
}


template<class TIME>
ref<Object>
PyTime<TIME>::method_get_parts(
  PyTime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"time_zone", nullptr};
  Object* tz;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &tz);

  auto parts = get_parts(self->time_, *convert_to_time_zone(tz));

  auto ymd_date = make_ymd_date(
    ora::YmdDate{parts.date.year, parts.date.month, parts.date.day});  // FIXME
  auto hms_daytime = make_hms_daytime(parts.daytime);

  auto time_zone_parts = get_time_zone_parts_type()->New();
  time_zone_parts->initialize(0, Long::FromLong(parts.time_zone.offset));
  time_zone_parts->initialize(1, Unicode::from(parts.time_zone.abbreviation));
  time_zone_parts->initialize(2, Bool::from(parts.time_zone.is_dst));

  auto time_parts = get_time_parts_type()->New();
  time_parts->initialize(0, std::move(ymd_date));
  time_parts->initialize(1, std::move(hms_daytime));
  time_parts->initialize(2, std::move(time_zone_parts));

  return std::move(time_parts);
}


template<class TIME>
Methods<PyTime<TIME>>
PyTime<TIME>::tp_methods_
  = Methods<PyTime>()
    .template add<method___format__>                    ("__format__")
    .template add<method_format>                        ("format",      docstring::pytime::format)
    .template add_class<method_from_offset>             ("from_offset")
    .template add<method_get_parts>                     ("get_parts")
  ;


//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

template<class TIME>
ref<Object>
PyTime<TIME>::get_invalid(
  PyTime* const self,
  void* /* closure */)
{
  return Bool::from(self->time_.is_invalid());
}


template<class TIME>
ref<Object>
PyTime<TIME>::get_missing(
  PyTime* const self,
  void* /* closure */)
{
  return Bool::from(self->time_.is_missing());
}


template<class TIME>
ref<Object>
PyTime<TIME>::get_offset(
  PyTime* const self,
  void* /* closure */)
{
  return Long::from(self->time_.get_offset());
}


template<class TIME>
ref<Object>
PyTime<TIME>::get_std(
  PyTime* const self,
  void* /* closure */)
{
  if (!self->time_.is_valid())
    throw py::ValueError("time not valid");

  auto const local = 
    ora::to_local<ora::Date, ora::UsecDaytime>(self->time_, *ora::UTC);
  auto const ymd = ora::date::get_ymd(local.date);
  auto const usec = local.daytime.get_offset();

  // FIXME: Maybe this should go elsewhere?
  static auto timezone_type = import("datetime", "timezone");
  static auto utc_obj = timezone_type->GetAttrString("utc");

  if (PyDateTimeAPI == nullptr)
    PyDateTime_IMPORT;
  return ref<Object>::take(
    PyDateTimeAPI->DateTime_FromDateAndTime(
      ymd.year,
      ymd.month,
      ymd.day,
      usec               / 3600000000u,
      usec % 3600000000u /   60000000u,
      usec %   60000000u /    1000000u,
      usec %    1000000u,
      utc_obj,
      PyDateTimeAPI->DateTimeType));
}


template<class TIME>
ref<Object>
PyTime<TIME>::get_valid(
  PyTime* const self,
  void* /* closure */)
{
  return Bool::from(self->time_.is_valid());
}


template<class TIME>
GetSets<PyTime<TIME>>
PyTime<TIME>::tp_getsets_ 
  = GetSets<PyTime>()
    .template add_get<get_invalid>      ("invalid")
    .template add_get<get_missing>      ("missing")
    .template add_get<get_offset>       ("offset")
    .template add_get<get_std>          ("std")
    .template add_get<get_valid>        ("valid")
  ;


//------------------------------------------------------------------------------
// Other members
//------------------------------------------------------------------------------

template<class TIME>
unique_ptr<ora::time::TimeFormat>
PyTime<TIME>::repr_format_;


template<class TIME>
unique_ptr<ora::time::TimeFormat>
PyTime<TIME>::str_format_;


//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

template<class TIME>
Type
PyTime<TIME>::build_type(
  string const& type_name)
{
  // Customize the type docstring with this class's name and parameters.
  auto const doc_len    = strlen(docstring::pytime::type) + 64;
  auto const doc        = new char[doc_len];
  auto const dot        = type_name.find_last_of('.');
  auto unqualified_name = 
    dot == string::npos ? type_name : type_name.substr(dot + 1);
  snprintf(
    doc, doc_len, docstring::pytime::type,
    unqualified_name.c_str(),
    to_string(TIME::MIN).c_str(), to_string(TIME::MAX).c_str());

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
    (hashfunc)            wrap<PyTime, tp_hash>,          // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            wrap<PyTime, tp_str>,           // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         doc,                            // tp_doc
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

using PyTimeDefault = PyTime<ora::time::Time>;

template<class TIME>
inline TIME
localtime_to_time(
  Sequence* const parts)
{
  assert(parts->Length() == 2);
  auto const localtime  = cast<Sequence>(parts->GetItem(0));
  auto const dd = to_datenum_daytick(localtime);
  auto const tz = convert_to_time_zone(parts->GetItem(1));
  return ora::from_local<TIME>(dd.first, dd.second, *tz);
}


template<class TIME>
inline TIME
date_daytime_to_time(
  Sequence* const parts)
{
  assert(parts->Length() == 3);
  auto const datenum    = to_datenum(parts->GetItem(0));
  auto const daytick    = to_daytick(parts->GetItem(1));
  auto const tz         = convert_to_time_zone(parts->GetItem(2));
  return ora::from_local<TIME>(datenum, daytick, *tz);
}


template<class TIME>
inline TIME
parts_to_time(
  Sequence* const parts)
{
  assert(parts->Length() == 7);
  auto const year   = parts->GetItem(0)->long_value();
  auto const month  = parts->GetItem(1)->long_value();
  auto const day    = parts->GetItem(2)->long_value();
  auto const hour   = parts->GetItem(3)->long_value();
  auto const minute = parts->GetItem(4)->long_value();
  auto const second = parts->GetItem(5)->double_value();
  auto const tz     = convert_to_time_zone(parts->GetItem(6));
  return 
    ora::from_local_parts<TIME>(year, month, day, hour, minute, second, *tz);
}


template<class TIME>
std::pair<bool, TIME>
maybe_time(
  Object* const obj)
{
  // An object of the same type?
  if (PyTime<TIME>::Check(obj))
    return {true, cast<PyTime<TIME>>(obj)->time_};

  // A different instance of the time class?
  auto const api = PyTimeAPI::get(obj);
  if (api != nullptr) 
    return {true, TIME(api->get_time128(obj))};

  // A 'datetime.datetime'?
  if (PyDateTimeAPI == nullptr)
    PyDateTime_IMPORT;
  if (PyDateTime_Check(obj)) {
    // First, make sure it's localized.
    auto const tzinfo = obj->GetAttrString("tzinfo", false);
    if (tzinfo == None)
      throw py::ValueError("unlocalized datetime doesn't represent a time");
    auto const tz = maybe_time_zone(tzinfo);
    if (tz == nullptr)
      throw py::ValueError(
        string("unknown tzinfo: ") + tzinfo->Repr()->as_utf8_string());
    
    // FIXME: Provide a all-integer ctor with (sec, usec).
    auto const time = ora::from_local_parts<TIME>(
      PyDateTime_GET_YEAR(obj),
      PyDateTime_GET_MONTH(obj) - 1,
      PyDateTime_GET_DAY(obj) - 1,
      PyDateTime_DATE_GET_HOUR(obj),
      PyDateTime_DATE_GET_MINUTE(obj),
      PyDateTime_DATE_GET_SECOND(obj) 
      + PyDateTime_DATE_GET_MICROSECOND(obj) * 1e-6,
      *tz,
      true);
    return {true, time};
  }

  // No type match.
  return {false, TIME::INVALID};
}


template<class TIME>
TIME
convert_to_time(
  Object* const obj)
{
  if (obj == None)
    // Use the default value.
    return TIME{};

  auto time = maybe_time<TIME>(obj);
  if (time.first)
    return time.second;

  if (Sequence::Check(obj)) {
    auto const parts = cast<Sequence>(obj);
    auto const length = parts->Length();
    if (length == 2)
      return localtime_to_time<TIME>(parts);
    else if (length == 3)
      return date_daytime_to_time<TIME>(parts);
    else if (length == 7)
      return parts_to_time<TIME>(parts);
  }

  // FIXME: Parse strings.

  throw py::TypeError("can't convert to a time: "s + *obj->Repr());
}


//------------------------------------------------------------------------------

#ifdef __clang__
// Use explicit instantiation for the main instances.
// FIXME: GCC 5.2.1 generates PyTime<>::type_ in BSS, which breaks linking.
extern template class PyTime<ora::time::Time>;
extern template class PyTime<ora::time::SmallTime>;
extern template class PyTime<ora::time::NsecTime>;
extern template class PyTime<ora::time::Unix32Time>;
extern template class PyTime<ora::time::Unix64Time>;
extern template class PyTime<ora::time::Time128>;
#endif

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora


