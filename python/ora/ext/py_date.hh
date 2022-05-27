#pragma once
#pragma GCC diagnostic ignored "-Wparentheses"

#include <cstring>
#undef _LIBCPP_WARN_ON_DEPRECATED_EXPERIMENTAL_HEADER
#include <iostream>
#include <optional>
#include <memory>
#include <string>
#include <unordered_map>

#include <Python.h>
#include <datetime.h>

#include "np.hh"
#include "ora.hh"
#include "py.hh"
#include "py_local.hh"
#include "types.hh"

namespace ora {
namespace py {

using namespace std::literals;

using std::make_unique;
using std::optional;
using std::string;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

extern ref<Object> make_ordinal_date(ora::OrdinalDate);
extern ref<Object> make_week_date(ora::WeekDate);
extern ref<Object> make_ymd_date(ora::YmdDate);

extern ref<Object> get_month_obj(int month);
extern ref<Object> get_weekday_obj(int weekday);

extern Weekday convert_to_weekday(Object*);

extern ref<Object> to_daytime_object(Object* obj);

/*
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
template<class DATE=Date> optional<DATE> maybe_date(Object*);

/*
 * Converts various kinds of Python objects to Date.
 *
 * If 'obj' can be converted unambiguously to a date, returns it.  Otherwise,
 * raises a Python exception.
 */
template<class DATE=Date> DATE convert_to_date(Object*);

/**
 * Converts various kinds of Python objects to a Date object, if possible.
 *
 * If `obj` cannot be converted to a date, returns a null reference.
 */
extern ref<Object> to_daytime_object(Object* obj);

/*
 * Helper for converting a 2-element sequence of ordinal date parts.
 */
template<class DATE=Date> inline DATE ordinal_date_to_date(Sequence*);

/*
 * Helper for converting a 3-element sequence of week date parts.
 */
template<class DATE=Date> inline DATE week_date_to_date(Sequence*);

/*
 * Helper for converting a 3-element sequence of date parts.
 */
template<class DATE=Date> inline DATE ymd_to_date(Sequence*);

//------------------------------------------------------------------------------
// Virtual API
//------------------------------------------------------------------------------

/*
 * Provides an API with dynamic dispatch to PyDate objects.
 *
 * The PyDate class, since it is a Python type, cannot be a virtual C++ class.
 * This mechanism interfaces the C++ virtual method mechanism with the Python
 * type system by mapping the Python type to a stub virtual C++ class.
 */
class PyDateAPI
{
public:

  virtual ~PyDateAPI() {}

  /*
   * Registers a virtual API for a Python type.
   */
  static PyDateAPI* add(
    PyTypeObject* const type, 
    std::unique_ptr<PyDateAPI>&& api)
  {
    return apis_.emplace(type, std::move(api)).first->second.get();
  }

  /*
   * Returns the API for a Python object, or nullptr if it isn't a PyDate.
   */
  static PyDateAPI const*
  get(
    PyTypeObject* const type)
  {
    auto api = apis_.find(type);
    return api == apis_.end() ? nullptr : api->second.get();
  }

  static PyDateAPI const* get(PyObject* const obj)
    { return get(obj->ob_type); }

  static PyDateAPI const* 
  from(
    PyObject* const obj)
  {
    auto const api = get(obj);
    if (api == nullptr) 
      throw TypeError("not an ora date type");
    else
      return api;
  }

  // API methods.
  virtual ora::Datenum              get_datenum(Object* date) const = 0;
  virtual ref<Object>               from_datenum(ora::Datenum) const = 0;
  virtual ref<Object>               from_parts(FullDate) const =0 ;
  virtual bool                      is_invalid(Object* time) const = 0;
  virtual bool                      is_missing(Object* time) const = 0;
  virtual ref<Object>               today(ora::TimeZone const& tz) const = 0;

private:

  static std::unordered_map<PyTypeObject*, std::unique_ptr<PyDateAPI>> apis_;

};


//------------------------------------------------------------------------------
// Docstrings
//------------------------------------------------------------------------------

namespace docstring {
namespace pydate {

#include "py_date.docstrings.hh.inc"

}  // namespace docstring
}  // namespace pydate


//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

/*
 * Template for a Python extension type wrapping a date class.
 *
 * 'DATE' is the wrapped date class, an instance of DateTemplate.  Invoke
 * add_to() to construct the type's PyTypeObject, ready it, and add it to a
 * module.
 */
template<class DATE>
class PyDate
  : public ExtensionType
{
public:

  using Date = DATE;

  /* 
   * Readies the Python type and adds it to `module` as `name`.  
   *
   * Should only be called once; this is not checked.
   */
  static void add_to(Module& module, string const& name);

  /*
   * Creates an instance of the Python type.
   */
  static ref<PyDate> create(Date date, PyTypeObject* type=&type_);

  /*
   * Returns true if 'object' is an instance of this type.
   */
  static bool Check(PyObject* object);

  static std::string repr(Date const date) 
    { return (*repr_format_)(date); }

  PyDate(Date date) : date_(date) {}

  /*
   * The wrapped date instance.
   *
   * This is the only non-static data member.
   */
  Date const date_;

private:

  class API
  : public PyDateAPI
  {
  public:
    
    virtual ora::Datenum get_datenum(Object* const date) const
      { return ((PyDate*) date)->date_.get_datenum(); }
    virtual ref<Object> from_datenum(ora::Datenum const datenum) const
      { return PyDate::create(ora::date::from_datenum<Date>(datenum)); }
    virtual ref<Object> from_parts(ora::FullDate const parts) const
      { return PyDate::create(ora::date::from_datenum(parts_to_datenum(parts))); }
    virtual bool is_invalid(Object* const date) const
      { return ((PyDate*) date)->date_.is_invalid(); }
    virtual bool is_missing(Object* const date) const
      { return ((PyDate*) date)->date_.is_missing(); }
    virtual ref<Object> today(ora::TimeZone const& tz) const
      { return PyDate::create(ora::today(tz)); }

  };

  static void           tp_init(PyDate* self, Tuple* args, Dict* kw_args);
  static void           tp_dealloc(PyDate* self);
  static ref<Unicode>   tp_repr(PyDate* self);
  static Py_hash_t      tp_hash(PyDate* self);
  static ref<Unicode>   tp_str(PyDate* self);
  static ref<Object>    tp_richcompare(PyDate* self, Object* other, int comparison);

  // Number methods.
  static ref<Object> nb_add         (PyDate* self, Object* other, bool right);
  static ref<Object> nb_subtract    (PyDate* self, Object* other, bool right);
  static ref<Object> nb_int         (PyDate* self);
  static ref<Object> nb_float       (PyDate* self);
  static PyNumberMethods tp_as_number_;

  // Methods.
  static ref<Object> method___format__          (PyDate*, Tuple*, Dict*);
  static ref<Object> method_from_datenum        (PyTypeObject*, Tuple*, Dict*);
  static ref<Object> method_from_iso_date       (PyTypeObject*, Tuple*, Dict*);
  static ref<Object> method_from_offset         (PyTypeObject*, Tuple*, Dict*);
  static ref<Object> method_from_ordinal_date   (PyTypeObject*, Tuple*, Dict*);
  static ref<Object> method_from_ymd            (PyTypeObject*, Tuple*, Dict*);
  static ref<Object> method_from_week_date      (PyTypeObject*, Tuple*, Dict*);
  static ref<Object> method_from_ymdi           (PyTypeObject*, Tuple*, Dict*);
  static Methods<PyDate> tp_methods_;

  // Getsets.
  static ref<Object> get_datenum                (PyDate*, void*);
  static ref<Object> get_day                    (PyDate*, void*);
  static ref<Object> get_invalid                (PyDate*, void*);
  static ref<Object> get_missing                (PyDate*, void*);
  static ref<Object> get_month                  (PyDate*, void*);
  static ref<Object> get_offset                 (PyDate*, void*);
  static ref<Object> get_ordinal                (PyDate*, void*);
  static ref<Object> get_ordinal_date           (PyDate*, void*);
  static ref<Object> get_std                    (PyDate*, void*);
  static ref<Object> get_valid                  (PyDate*, void*);
  static ref<Object> get_week                   (PyDate*, void*);
  static ref<Object> get_week_date              (PyDate*, void*);
  static ref<Object> get_week_year              (PyDate*, void*);
  static ref<Object> get_weekday                (PyDate*, void*);
  static ref<Object> get_year                   (PyDate*, void*);
  static ref<Object> get_ymd                    (PyDate*, void*);
  static ref<Object> get_ymdi                   (PyDate*, void*);
  static GetSets<PyDate> tp_getsets_;

  // Date format used to generate the repr.
  static unique_ptr<ora::date::DateFormat> repr_format_;

  static Type build_type(string const& type_name);

public:

  static Type type_;
  static PyDateAPI const* api_;

};


template<class DATE>
void
PyDate<DATE>::add_to(
  Module& module,
  string const& name)
{
  // Construct the type struct.
  type_ = build_type(string{module.GetName()} + "." + name);
  // FIXME: Make the conditional on successfully importing numpy.
  type_.tp_base = &PyGenericArrType_Type;
  // Hand it to Python.
  type_.Ready();

  // Set up the API.
  api_ = PyDateAPI::add(&type_, std::make_unique<API>());

  // Build the repr format.
  repr_format_ = make_unique<ora::date::DateFormat>(
    name + "(%0Y, %~B, %0d)",
    name + ".INVALID",
    name + ".MISSING");

  // Add in static data members.
  Dict* const dict = (Dict*) type_.tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("EPOCH"   , create(Date::from_offset(0)));
  dict->SetItemString("INVALID" , create(Date::INVALID));
  dict->SetItemString("MAX"     , create(Date::MAX));
  dict->SetItemString("MIN"     , create(Date::MIN));
  dict->SetItemString("MISSING" , create(Date::MISSING));

  // Add the type to the module.
  module.add(&type_);
}


template<class DATE>
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


template<class DATE>
bool
PyDate<DATE>::Check(
  PyObject* other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

template<class DATE>
void
PyDate<DATE>::tp_init(
  PyDate* const self, 
  Tuple* const args, 
  Dict* const kw_args)
{
  if (kw_args != nullptr && kw_args->Size() > 0)
    throw TypeError("function takes no keyword arguments");
  auto const num_args = args->Length();
  Date date;
  if (num_args == 0)
    ;
  else if (num_args == 1)
    date = convert_to_date<Date>(args->GetItem(0));
  else if (num_args == 2)
    date = ordinal_date_to_date<Date>(args);
  else if (num_args == 3)
    date = ymd_to_date<Date>(args);
  else
    throw TypeError("function takes 0, 1, 2, or 3 arguments");

  new(self) PyDate{date};
}


template<class DATE>
void
PyDate<DATE>::tp_dealloc(
  PyDate* const self)
{
  self->date_.~DateTemplate();
  self->ob_type->tp_free(self);
}


template<class DATE>
ref<Unicode>
PyDate<DATE>::tp_repr(
  PyDate* const self)
{
  return Unicode::from(repr(self->date_));
}


template<class DATE>
Py_hash_t
PyDate<DATE>::tp_hash(
  PyDate* const self)
{
  return 
      self->date_.is_invalid() ? std::numeric_limits<Py_hash_t>::max()
    : self->date_.is_missing() ? std::numeric_limits<Py_hash_t>::max() - 1
    : self->date_.get_offset();
}


template<class DATE>
ref<Unicode>
PyDate<DATE>::tp_str(
  PyDate* const self)
{
  // FIXME: Make the format configurable.
  auto& format = ora::date::DateFormat::DEFAULT;
  return Unicode::from(format(self->date_));
}


template<class DATE>
ref<Object>
PyDate<DATE>::tp_richcompare(
  PyDate* const self,
  Object* const other,
  int const comparison)
{
  auto const other_date = maybe_date<Date>(other);
  if (!other_date)
    return not_implemented_ref();
  return richcmp(self->date_, *other_date, comparison);
}


//------------------------------------------------------------------------------
// Number methods
//------------------------------------------------------------------------------

template<class DATE>
inline ref<Object>
PyDate<DATE>::nb_add(
  PyDate* const self,
  Object* const other,
  bool /* ignored */)
{
  auto offset = other->maybe_long_value();
  if (offset)
    if (*offset == 0)
      return ref<PyDate>::of(self);
    else 
      return create(self->date_ + *offset, self->ob_type);
  else
    return not_implemented_ref();
}


template<class DATE>
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
    return Long::FromLong(self->date_ - *other_date);

  auto offset = other->maybe_long_value();
  if (offset)
    return 
        *offset == 0 
      ? ref<PyDate>::of(self) 
      : create(self->date_ - *offset, self->ob_type);

  return not_implemented_ref();
}


template<class DATE>
inline ref<Object>
PyDate<DATE>::nb_int(
  PyDate* const self)
{
  throw TypeError("int() argument cannot be a date");
}


template<class DATE>
inline ref<Object>
PyDate<DATE>::nb_float(
  PyDate* const self)
{
  throw TypeError("float() argument cannot be a date");
}


template<class DATE>
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
  // Work around a NumPy bug (https://github.com/numpy/numpy/issues/10693) by
  // defining nb_int, nb_float that raise TypeError.
  (unaryfunc)   wrap<PyDate, nb_int>,           // nb_int
  (void*)       nullptr,                        // nb_reserved
  (unaryfunc)   wrap<PyDate, nb_float>,         // nb_float
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
  (binaryfunc)  nullptr,                        // nb_matrix_multiply
  (binaryfunc)  nullptr,                        // nb_inplace_matrix_multiply
};


//------------------------------------------------------------------------------
// Methods
//------------------------------------------------------------------------------

template<class DATE>
ref<Object>
PyDate<DATE>::method___format__(
  PyDate* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  if (args->GetLength() != 1 || kw_args != nullptr)
    throw TypeError("__format__() takes one argument");
  auto const pattern = args->GetItem(0)->Str()->as_utf8();
  auto const result = 
    // Empty pattern?  Use the default formatter.
    strlen(pattern) == 0 ? ora::date::DateFormat::DEFAULT(self->date_)
    : ora::DateFormat(pattern)(self->date_);
  return Unicode::from(result);
}


template<class DATE>
ref<Object>
PyDate<DATE>::method_from_datenum(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"datenum", nullptr};
  ora::Datenum datenum;
  static_assert(
    sizeof(ora::Datenum) == sizeof(int),
    "datenum is not an int");
  Arg::ParseTupleAndKeywords(args, kw_args, "i", arg_names, &datenum);

  return create(ora::date::from_datenum<Date>(datenum), type);
}


template<class DATE>
ref<Object>
PyDate<DATE>::method_from_iso_date(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"iso_date", nullptr};
  char* iso_date;
  Arg::ParseTupleAndKeywords(args, kw_args, "s", arg_names, &iso_date);

  return create(ora::date::from_iso_date<Date>(iso_date), type);
}


template<class DATE>
ref<Object>
PyDate<DATE>::method_from_offset(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  // Using long is probably OK for dates. (?)
  static char const* const arg_names[] = {"offset", nullptr};
  long offset;
  Arg::ParseTupleAndKeywords(args, kw_args, "k", arg_names, &offset);

  return create(ora::date::from_offset<Date>(offset), type);
}


template<class DATE>
ref<Object>
PyDate<DATE>::method_from_ordinal_date(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("from_ordinal_date() takes no keyword arguments");

  auto const num_args = args->Length();
  Sequence* parts;
  // Accept either a single two-element sequence, or two args.
  if (num_args == 1) {
    parts = cast<Sequence>(args->GetItem(0));
    if (parts->Length() != 2)
      throw TypeError("arg must be a 2-element sequence");
  }
  else if (num_args == 2)
    parts = args;
  else
    throw TypeError("from_week_date() takes 1 or 2 args");

  return create(ordinal_date_to_date<Date>(parts), type);
}


template<class DATE>
ref<Object>
PyDate<DATE>::method_from_week_date(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("from_week_date() takes no keyword arguments");

  auto const num_args = args->Length();
  Sequence* parts;
  // Accept either a single three-element sequence, or three args.
  if (num_args == 1) {
    parts = cast<Sequence>(args->GetItem(0));
    if (parts->Length() != 3)
      throw TypeError("arg must be a 3-element sequence");
  }
  else if (num_args == 3)
    parts = args;
  else
    throw TypeError("from_week_date() takes 1 or 3 args");

  return create(week_date_to_date<Date>(parts), type);
}


template<class DATE>
ref<Object>
PyDate<DATE>::method_from_ymd(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("from_ymd() takes no keyword arguments");

  auto const num_args = args->Length();
  Sequence* parts;
  // Accept either a single three-element sequence, or three args.
  if (num_args == 1) {
    parts = cast<Sequence>(args->GetItem(0));
    if (parts->Length() != 3)
      throw TypeError("arg must be a 3-element sequence");
  }
  else if (num_args == 3)
    parts = args;
  else
    throw TypeError("from_ymd() takes one or three arguments");

  return create(ymd_to_date<Date>(parts), type);
}


template<class DATE>
ref<Object>
PyDate<DATE>::method_from_ymdi(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"ymdi", nullptr};
  int ymdi;
  Arg::ParseTupleAndKeywords(args, kw_args, "i", arg_names, &ymdi);

  return create(ora::date::from_ymdi<DATE>(ymdi), type);
}


template<class DATE>
Methods<PyDate<DATE>>
PyDate<DATE>::tp_methods_
  = Methods<PyDate>()
    .template add<method___format__>                ("__format__")
    .template add_class<method_from_datenum>        ("from_datenum",        docstring::pydate::from_datenum)
    .template add_class<method_from_iso_date>       ("from_iso_date",       docstring::pydate::from_iso_date)
    .template add_class<method_from_offset>         ("from_offset",         docstring::pydate::from_offset)
    .template add_class<method_from_ordinal_date>   ("from_ordinal_date",   docstring::pydate::from_ordinal_date)
    .template add_class<method_from_week_date>      ("from_week_date",      docstring::pydate::from_week_date)
    .template add_class<method_from_ymd>            ("from_ymd",            docstring::pydate::from_ymd)
    .template add_class<method_from_ymdi>           ("from_ymdi",           docstring::pydate::from_ymdi)
  ;


//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

template<class DATE>
ref<Object>
PyDate<DATE>::get_datenum(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(self->date_.get_datenum());
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_day(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(ora::date::get_ymd(self->date_).day);
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_invalid(
  PyDate* const self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_invalid());
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_missing(
  PyDate* const self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_missing());
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_month(
  PyDate* const self,
  void* /* closure */)
{
  return get_month_obj(ora::date::get_ymd(self->date_).month);
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_offset(
  PyDate* const self,
  void* /* closure */)
{
  return Long::from(self->date_.get_offset());
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_ordinal(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(ora::date::get_ordinal_date(self->date_).ordinal);
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_ordinal_date(
  PyDate* const self,
  void* /* closure */)
{
  return make_ordinal_date(ora::date::get_ordinal_date(self->date_));
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_std(
  PyDate* const self,
  void* /* closure */)
{
  if (!self->date_.is_valid())
    throw py::ValueError("date not valid");

  if (PyDateTimeAPI == nullptr)
    PyDateTime_IMPORT;
  auto const ymd = ora::date::get_ymd(self->date_);
  return ref<Object>::take(PyDate_FromDate(ymd.year, ymd.month, ymd.day));
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_valid(
  PyDate* const self,
  void* /* closure */)
{
  return Bool::from(self->date_.is_valid());
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_week(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(ora::date::get_week_date(self->date_).week);
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_week_date(
  PyDate* const self,
  void* /* closure */)
{
  return make_week_date(ora::date::get_week_date(self->date_));
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_week_year(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(ora::date::get_week_date(self->date_).week_year);
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_weekday(
  PyDate* const self,
  void* /* closure */)
{
  return get_weekday_obj(ora::date::get_weekday(self->date_));
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_year(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(ora::date::get_ordinal_date(self->date_).year);
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_ymd(
  PyDate* const self,
  void* /* closure */)
{
  return make_ymd_date(ora::date::get_ymd(self->date_));
}


template<class DATE>
ref<Object>
PyDate<DATE>::get_ymdi(
  PyDate* const self,
  void* /* closure */)
{
  return Long::FromLong(ora::date::get_ymdi(self->date_));
}


template<class DATE>
GetSets<PyDate<DATE>>
PyDate<DATE>::tp_getsets_ 
  = GetSets<PyDate>()
    .template add_get<get_datenum>      ("datenum"      , docstring::pydate::datenum)
    .template add_get<get_day>          ("day"          , docstring::pydate::day)
    .template add_get<get_invalid>      ("invalid"      , docstring::pydate::invalid)
    .template add_get<get_missing>      ("missing"      , docstring::pydate::missing)
    .template add_get<get_month>        ("month"        , docstring::pydate::month)
    .template add_get<get_offset>       ("offset"       , docstring::pydate::offset)
    .template add_get<get_ordinal>      ("ordinal"      , docstring::pydate::ordinal)
    .template add_get<get_ordinal_date> ("ordinal_date" , docstring::pydate::ordinal_date)
    .template add_get<get_std>          ("std"          , docstring::pydate::std)
    .template add_get<get_valid>        ("valid"        , docstring::pydate::valid)
    .template add_get<get_week>         ("week"         , docstring::pydate::week)
    .template add_get<get_week_date>    ("week_date"    , docstring::pydate::week_date)
    .template add_get<get_week_year>    ("week_year"    , docstring::pydate::week_year)
    .template add_get<get_weekday>      ("weekday"      , docstring::pydate::weekday)
    .template add_get<get_year>         ("year"         , docstring::pydate::year)
    .template add_get<get_ymdi>         ("ymdi"         , docstring::pydate::ymdi)
    .template add_get<get_ymd>          ("ymd"          , docstring::pydate::ymd)
  ;


//------------------------------------------------------------------------------
// Other members
//------------------------------------------------------------------------------

template<class DATE>
unique_ptr<ora::date::DateFormat>
PyDate<DATE>::repr_format_;


//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

template<class DATE>
Type
PyDate<DATE>::build_type(
  string const& type_name)
{
  // Customize the type docstring with this class's name and parameters.
  auto const doc_len    = strlen(docstring::pydate::type) + 64;
  auto const doc        = new char[doc_len];
  auto const dot        = type_name.find_last_of('.');
  auto unqualified_name = 
    dot == string::npos ? type_name : type_name.substr(dot + 1);
  snprintf(
    doc, doc_len, docstring::pydate::type,
    unqualified_name.c_str(),
    to_string(DATE::MIN).c_str(), to_string(DATE::MAX).c_str());

  return PyTypeObject{
    PyVarObject_HEAD_INIT(nullptr, 0)
    (char const*)         strdup(type_name.c_str()),      // tp_name
    (Py_ssize_t)          sizeof(PyDate),                 // tp_basicsize
    (Py_ssize_t)          0,                              // tp_itemsize
    (destructor)          wrap<PyDate, tp_dealloc>,       // tp_dealloc
    (printfunc)           nullptr,                        // tp_print
    (getattrfunc)         nullptr,                        // tp_getattr
    (setattrfunc)         nullptr,                        // tp_setattr
                          nullptr,                        // tp_reserved
    (reprfunc)            wrap<PyDate, tp_repr>,          // tp_repr
    (PyNumberMethods*)    &tp_as_number_,                 // tp_as_number
    (PySequenceMethods*)  nullptr,                        // tp_as_sequence
    (PyMappingMethods*)   nullptr,                        // tp_as_mapping
    (hashfunc)            wrap<PyDate, tp_hash>,          // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            wrap<PyDate, tp_str>,           // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         doc,                            // tp_doc
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


template<class DATE>
Type
PyDate<DATE>::type_;


template<class DATE>
PyDateAPI const*
PyDate<DATE>::api_;


//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

using PyDateDefault = PyDate<ora::date::Date>;

inline ref<Object>
make_date(
  ora::Datenum const datenum,
  PyTypeObject* type=&PyDateDefault::type_)
{
  // Special case fast path for the default date type.
  if (type == &PyDateDefault::type_)
    return PyDateDefault::create(
      ora::date::from_datenum<PyDateDefault::Date>(datenum));

  auto const api = PyDateAPI::get(type);
  if (api == nullptr)
    throw TypeError("not a date type: "s + *(((Object*) type)->Repr()));
  else
    return api->from_datenum(datenum);
}


template<class DATE>
inline DATE
ordinal_date_to_date(
  Sequence* const parts)
{
  long const year       = parts->GetItem(0)->long_value();
  long const ordinal    = parts->GetItem(1)->long_value();
  return ora::date::from_ordinal_date<DATE>(year, ordinal);
}


template<class DATE>
inline DATE
week_date_to_date(
  Sequence* const parts)
{
  long const week_year  = parts->GetItem(0)->long_value();
  long const week       = parts->GetItem(1)->long_value();
  long const weekday    = parts->GetItem(2)->long_value();
  return ora::date::from_week_date<DATE>(week_year, week, weekday);
}


template<class DATE>
inline DATE
ymd_to_date(
  Sequence* const parts)
{
  long const year   = parts->GetItem(0)->long_value();
  long const month  = parts->GetItem(1)->long_value();
  long const day    = parts->GetItem(2)->long_value();
  return ora::date::from_ymd<DATE>(year, month, day);
}


template<class DATE>
inline optional<DATE>
maybe_date(
  Object* const obj)
{
  // Try for an instance of the same PyDate.
  if (PyDate<DATE>::Check(obj)) 
    return static_cast<PyDate<DATE>*>(obj)->date_;

  // Try for an instance of a different PyDate template instance.
  auto const api = PyDateAPI::get(obj);
  if (api != nullptr) 
    return 
        api->is_invalid(obj) ? DATE::INVALID
      : api->is_missing(obj) ? DATE::MISSING
      : ora::date::from_datenum<DATE>(api->get_datenum(obj));

  // Try for datetime.date.  Note that PyDateTimeAPI is declared to be static,
  // so we have to initialize it in each compilation unit.
  if (PyDateTimeAPI == nullptr)
    PyDateTime_IMPORT;
  if (PyDate_Check(obj)) 
    return ora::date::from_ymd<DATE>(
      PyDateTime_GET_YEAR(obj),
      PyDateTime_GET_MONTH(obj),
      PyDateTime_GET_DAY(obj));

  // Try for a date type that as a toordinal() method, to handle duck typing
  // for datetime.date.
  auto ordinal = obj->CallMethodObjArgs("toordinal", false);
  if (ordinal != nullptr)
    return ora::date::from_datenum<DATE>(ordinal->long_value());

  // Try for a date type that has a datenum attribute, to handle duck typing
  // for our PyDate.
  auto datenum = obj->GetAttrString("datenum", false);
  if (datenum != nullptr) 
    return ora::date::from_datenum<DATE>(datenum->long_value());

  // No type match.
  return std::nullopt;
}


template<class DATE>
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

  if (Unicode::Check(obj)) {
    auto const str = static_cast<Unicode*>(obj)->as_utf8_string();
    if (str == "MIN")
      return DATE::MIN;
    else if (str == "MAX")
      return DATE::MAX;

    try {
      return ora::date::from_iso_date<DATE>(str);
    }
    catch (ora::DateError const&) {
      throw py::ValueError("can't parse as date: '"s + str + "'");
    }
  }

  if (Sequence::Check(obj)) {
    auto const seq = static_cast<Sequence*>(obj);
    if (seq->Length() == 3) 
      // Interpret a three-element sequence as date parts.
      return ymd_to_date<DATE>(seq);
    else if (seq->Length() == 2) 
      // Interpret a two-element sequence as ordinal parts.
      try {
        return ordinal_date_to_date<DATE>(seq);
      }
      catch (ora::DateError const&) {
        throw py::ValueError("can't convert to a date: "s + *obj->Repr());
      }
  }

  auto const long_obj = obj->Long(false);
  if (long_obj != nullptr) {
    // Interpret eight-digit values as YMDI.
    long const ymdi = (long) *long_obj;
    if (10000000 <= ymdi && ymdi <= 99999999) 
      try {
        return ora::date::from_ymdi<DATE>(ymdi);
      }
      catch (ora::DateError const&) {
        throw py::ValueError(
          "can't convert to a date: "s + std::to_string(ymdi));
      }
  }

  throw py::TypeError("can't convert to a date: "s + *obj->Repr());
}


//------------------------------------------------------------------------------

#ifdef __clang__
// Use explicit instantiation for the main instances.
// FIXME: GCC 5.2.1 generates PyDate<>::type_ in BSS, which breaks linking.
extern template class PyDate<ora::date::Date>;
extern template class PyDate<ora::date::Date16>;
#endif

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

