#pragma once
#pragma GCC diagnostic ignored "-Wparentheses"

#include <cmath>
#include <experimental/optional>
#include <iostream>

#include <Python.h>
#include <datetime.h>

#include "cron/format.hh"
#include "cron/daytime.hh"
#include "cron/time_zone.hh"
#include "py.hh"

namespace aslib {

using namespace py;
using namespace std::literals;

using std::experimental::optional;
using std::make_unique;
using std::string;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

StructSequenceType* get_daytime_parts_type();

/**
 * Attempts to convert various kinds of Python daytime object to 'DAYTIME'.
 *
 * If 'obj' is a daytime object of some kind, returns the equivalent daytime;
 * otherwise a null option.  The following daytime objects are recognized:
 *
 *  - PyDaytimeTemplate instances
 *  - 'datetime.time` instances
 */
template<typename DAYTIME> optional<DAYTIME> maybe_daytime(Object*);

/**
 * Converts various kinds of Python objects to Daytime.
 *
 * If 'obj' can be converted unambiguously to a daytime, returns it; otherwise
 * rasies a Python exception.
 */
template<typename DAYTIME> DAYTIME convert_to_daytime(Object*);

/**
 * Helper for converting a 2- or 3-eleme4nt sequence of daytime parts.
 */
template<typename DAYTIME> inline DAYTIME parts_to_daytime(Sequence*);

//------------------------------------------------------------------------------
// Type class
//------------------------------------------------------------------------------

/**
 * Template for a Python extension type wrapping a daytime class.
 *
 * 'DAYTIME' is the wrapped daytime class, an instance of DaytimeTemplate.
 * Invoke add_to() to construct the type's PyTypeObject, ready it, and add it 
 * to a module.
 */
template<typename DAYTIME>
class PyDaytime
  : public ExtensionType
{
public:

  using Daytime = DAYTIME;

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
  static ref<PyDaytime> create(Daytime daytime, PyTypeObject* type=&type_);

  /**
   * Returns true if 'object' is an instance of this type.
   */
  static bool Check(PyObject* object);

  PyDaytime(Daytime daytime) : daytime_(daytime) {}

  /**
   * The wrapped date instance.
   *
   * This is the only non-static data member.
   */
  Daytime const daytime_;

private:

  static void tp_init(PyDaytime* self, Tuple* args, Dict* kw_args);
  static void tp_dealloc(PyDaytime* self);
  static ref<Unicode> tp_repr(PyDaytime* self);
  static ref<Unicode> tp_str(PyDaytime* self);
  static ref<Object> tp_richcompare(PyDaytime* self, Object* other, int comparison);

  // Number methods.
  static ref<Object> nb_add     (PyDaytime* self, Object* other, bool right);
  static ref<Object> nb_subtract(PyDaytime* self, Object* other, bool right);
  static PyNumberMethods tp_as_number_;

  // Methods.
  static ref<Object> method_from_daytick        (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_parts          (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_ssm            (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_is_same             (PyDaytime*    self, Tuple* args, Dict* kw_args);
  static Methods<PyDaytime> tp_methods_;

  // Getsets.
  static ref<Object> get_daytick                (PyDaytime* self, void*);
  static ref<Object> get_hour                   (PyDaytime* self, void*);
  static ref<Object> get_invalid                (PyDaytime* self, void*);
  static ref<Object> get_minute                 (PyDaytime* self, void*);
  static ref<Object> get_missing                (PyDaytime* self, void*);
  static ref<Object> get_parts                  (PyDaytime* self, void*);
  static ref<Object> get_second                 (PyDaytime* self, void*);
  static ref<Object> get_ssm                    (PyDaytime* self, void*);
  static ref<Object> get_valid                  (PyDaytime* self, void*);
  static GetSets<PyDaytime> tp_getsets_;

  /** Date format used to generate the repr and str.  */
  static unique_ptr<cron::DaytimeFormat> repr_format_;
  static unique_ptr<cron::DaytimeFormat> str_format_;

  static Type build_type(string const& type_name);

};


template<typename DAYTIME>
void
PyDaytime<DAYTIME>::add_to(
  Module& module,
  string const& name)
{
  // Construct the type struct.
  type_ = build_type(string{module.GetName()} + "." + name);
  // Hand it to Python.
  type_.Ready();

  // Build the repr format.
  repr_format_ = make_unique<cron::DaytimeFormat>(
    name + "(%0H, %0M, %0S)",
    name + ".INVALID",
    name + ".MISSING");

  // Build the str format.  Choose precision for seconds that captures actual
  // precision of the daytime class.
  std::string pattern = "%H:%M:%";
  size_t const precision = (size_t) ceil(log10(Daytime::DENOMINATOR));
  if (precision > 0) {
    pattern += ".";
    pattern += std::to_string(precision);
  }
  pattern += "S";
  str_format_ = make_unique<cron::DaytimeFormat>(pattern);

  // Add in static data members.
  auto const dict = (Dict*) type_.tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("EPSILON" , Float::FromDouble(1.0 / DAYTIME::DENOMINATOR));
  dict->SetItemString("INVALID" , create(Daytime::INVALID));
  dict->SetItemString("MAX"     , create(Daytime::MAX));
  dict->SetItemString("MIN"     , create(Daytime::MIN));
  dict->SetItemString("MISSING" , create(Daytime::MISSING));

  // Add the type to the module.
  module.add(&type_);
}


template<typename DAYTIME>
ref<PyDaytime<DAYTIME>>
PyDaytime<DAYTIME>::create(
  Daytime const daytime,
  PyTypeObject* const type)
{
  auto obj = ref<PyDaytime>::take(check_not_null(PyDaytime::type_.tp_alloc(type, 0)));

  // daytime_ is const to indicate immutablity, but Python initialization is
  // later than C++ initialization, so we have to cast off const here.
  new(const_cast<Daytime*>(&obj->daytime_)) Daytime{daytime};
  return obj;
}


template<typename DAYTIME>
Type
PyDaytime<DAYTIME>::type_;


template<typename DAYTIME>
bool
PyDaytime<DAYTIME>::Check(
  PyObject* const other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

template<typename DAYTIME>
void
PyDaytime<DAYTIME>::tp_init(
  PyDaytime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("function takes no keyword arguments");
  auto const num_args = args->Length();
  Daytime daytime;
  if (num_args == 0)
    ;
  else if (num_args == 1)
    daytime = convert_to_daytime<Daytime>(args->GetItem(0));
  else if (num_args == 2 || num_args == 3)
    daytime = parts_to_daytime<Daytime>(args);
  else
    throw TypeError("function takes 0, 1, 2, or 3 arguments");

  new(self) PyDaytime{daytime};
}


template<typename DAYTIME>
void
PyDaytime<DAYTIME>::tp_dealloc(
  PyDaytime* const self)
{
  self->daytime_.~DaytimeTemplate();
  self->ob_type->tp_free(self);
}


template<typename DAYTIME>
ref<Unicode>
PyDaytime<DAYTIME>::tp_repr(
  PyDaytime* const self)
{
  return Unicode::from((*repr_format_)(self->daytime_));
}


template<typename DAYTIME>
ref<Unicode>
PyDaytime<DAYTIME>::tp_str(
  PyDaytime* const self)
{
  return Unicode::from((*str_format_)(self->daytime_));  
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::tp_richcompare(
  PyDaytime* const self,
  Object* const other,
  int const comparison)
{
  auto const opt = maybe_daytime<Daytime>(other);
  if (!opt)
    return not_implemented_ref();
  
  Daytime const a0 = self->daytime_;
  Daytime const a1 = *opt;

  bool result;
  switch (comparison) {
  case Py_EQ: result = a0 == a1; break;
  case Py_GE: result = a0 >= a1; break;
  case Py_GT: result = a0 >  a1; break;
  case Py_LE: result = a0 <= a1; break;
  case Py_LT: result = a0 <  a1; break;
  case Py_NE: result = a0 != a1; break;
  default:    result = false; assert(false);
  }
  return Bool::from(result);
}


//------------------------------------------------------------------------------
// Number methods
//------------------------------------------------------------------------------

template<typename DAYTIME>
inline ref<Object>
PyDaytime<DAYTIME>::nb_add(
  PyDaytime* const self,
  Object* const other,
  bool /* ignored */)
{
  auto shift = other->maybe_double_value();
  if (shift)
    return 
      *shift == 0 ? ref<PyDaytime>::of(self)
      : create(self->daytime_ + *shift, self->ob_type);
  else
    return not_implemented_ref();
}


template<typename DAYTIME>
inline ref<Object>
PyDaytime<DAYTIME>::nb_subtract(
  PyDaytime* const self,
  Object* const other,
  bool right)
{
  if (right) 
    return not_implemented_ref();

  auto shift = other->maybe_double_value();
  if (shift)
    return 
      *shift == 0 ? ref<PyDaytime>::of(self)
      : create(self->daytime_ - *shift, self->ob_type);
  else
    return not_implemented_ref();
}


template<typename DAYTIME>
PyNumberMethods
PyDaytime<DAYTIME>::tp_as_number_ = {
  (binaryfunc)  wrap<PyDaytime, nb_add>,        // nb_add
  (binaryfunc)  wrap<PyDaytime, nb_subtract>,   // nb_subtract
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

template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::method_from_daytick(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"daytick", nullptr};
  cron::Daytick daytick;
  static_assert(
    sizeof(cron::Daytick) == sizeof(long), "daytick is not an long");
  Arg::ParseTupleAndKeywords(args, kw_args, "k", arg_names, &daytick);

  return create(Daytime::from_daytick(daytick), type);
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::method_from_parts(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("from_parts() takes no keyword arguments");

  Sequence* parts;
  // Accept either a single two- or three-element sequence, or three args.
  if (args->Length() == 1) {
    parts = cast<Sequence>(args->GetItem(0));
    if (parts->Length() < 3)
      throw TypeError("parts must be a 3-element (or longer) sequence");
  }
  else if (args->Length() == 2 || args->Length() == 3)
    parts = args;
  else
    throw TypeError("from_parts() takes one or three arguments");

  long   const hour   = parts->GetItem(0)->long_value();
  long   const minute = parts->GetItem(1)->long_value();
  double const second
    = args->Length() == 3 ? parts->GetItem(2)->double_value() : 0;
  return create(Daytime::from_parts(hour, minute, second), type);
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::method_from_ssm(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"ssm", nullptr};
  cron::Ssm ssm;
  Arg::ParseTupleAndKeywords(args, kw_args, "d", arg_names, &ssm);

  return create(Daytime::from_ssm(ssm), type);
}


// We call this method "is_same" because "is" is a keyword in Python.
template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::method_is_same(
  PyDaytime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"object", nullptr};
  Object* object;
  Arg::ParseTupleAndKeywords(args, kw_args, "O", arg_names, &object);

  auto daytime_opt = maybe_daytime<Daytime>(object);
  return Bool::from(daytime_opt && self->daytime_.is(*daytime_opt));
}


template<typename DAYTIME>
Methods<PyDaytime<DAYTIME>>
PyDaytime<DAYTIME>::tp_methods_
  = Methods<PyDaytime>()
    .template add_class<method_from_daytick>        ("from_daytick")
    .template add_class<method_from_parts>          ("from_parts")
    .template add_class<method_from_ssm>            ("from_ssm")
    .template add      <method_is_same>             ("is_same")
  ;

//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_daytick(
  PyDaytime* self,
  void* /* closure */)
{
  
  return Long::FromUnsignedLong(self->daytime_.get_daytick());
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_hour(
  PyDaytime* self,
  void* /* closure */)
{
  return Long::FromLong(self->daytime_.get_parts().hour);
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_invalid(
  PyDaytime* self,
  void* /* closure */)
{
  return Bool::from(self->daytime_.is_invalid());
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_minute(
  PyDaytime* self,
  void* /* closure */)
{
  return Long::FromLong(self->daytime_.get_parts().minute);
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_missing(
  PyDaytime* self,
  void* /* closure */)
{
  return Bool::from(self->daytime_.is_missing());
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_parts(
  PyDaytime* self,
  void* /* closure */)
{
  auto parts = self->daytime_.get_parts();
  auto parts_obj = get_daytime_parts_type()->New();
  parts_obj->initialize(0, Long::FromLong(parts.hour));
  parts_obj->initialize(1, Long::FromLong(parts.minute));
  parts_obj->initialize(2, Float::FromDouble(parts.second));
  return std::move(parts_obj);
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_second(
  PyDaytime* self,
  void* /* closure */)
{
  return Float::FromDouble(self->daytime_.get_parts().second);
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_ssm(
  PyDaytime* self,
  void* /* closure */)
{
  return Float::FromDouble(self->daytime_.get_ssm());
}


template<typename DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_valid(
  PyDaytime* self,
  void* /* closure */)
{
  return Bool::from(self->daytime_.is_valid());
}


template<typename DAYTIME>
GetSets<PyDaytime<DAYTIME>>
PyDaytime<DAYTIME>::tp_getsets_ 
  = GetSets<PyDaytime>()
    .template add_get<get_daytick>              ("daytick")
    .template add_get<get_hour>                 ("hour")
    .template add_get<get_invalid>              ("invalid")
    .template add_get<get_minute>               ("minute")
    .template add_get<get_missing>              ("missing")
    .template add_get<get_parts>                ("parts")
    .template add_get<get_second>               ("second")
    .template add_get<get_ssm>                  ("ssm")
    .template add_get<get_valid>                ("valid")
  ;


//------------------------------------------------------------------------------
// Other members
//------------------------------------------------------------------------------

template<typename DAYTIME>
unique_ptr<cron::DaytimeFormat>
PyDaytime<DAYTIME>::repr_format_;

template<typename DAYTIME>
unique_ptr<cron::DaytimeFormat>
PyDaytime<DAYTIME>::str_format_;

//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

template<typename DAYTIME>
Type
PyDaytime<DAYTIME>::build_type(
  string const& type_name)
{
  return PyTypeObject{
    PyVarObject_HEAD_INIT(nullptr, 0)
    (char const*)         strdup(type_name.c_str()),      // tp_name
    (Py_ssize_t)          sizeof(PyDaytime),              // tp_basicsize
    (Py_ssize_t)          0,                              // tp_itemsize
    (destructor)          wrap<PyDaytime, tp_dealloc>,    // tp_dealloc
    (printfunc)           nullptr,                        // tp_print
    (getattrfunc)         nullptr,                        // tp_getattr
    (setattrfunc)         nullptr,                        // tp_setattr
                          nullptr,                        // tp_reserved
    (reprfunc)            wrap<PyDaytime, tp_repr>,       // tp_repr
    (PyNumberMethods*)    &tp_as_number_,                 // tp_as_number
    (PySequenceMethods*)  nullptr,                        // tp_as_sequence
    (PyMappingMethods*)   nullptr,                        // tp_as_mapping
    (hashfunc)            nullptr,                        // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            wrap<PyDaytime, tp_str>,        // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         nullptr,                        // tp_doc
    (traverseproc)        nullptr,                        // tp_traverse
    (inquiry)             nullptr,                        // tp_clear
    (richcmpfunc)         wrap<PyDaytime, tp_richcompare>,// tp_richcompare
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
    (initproc)            wrap<PyDaytime, tp_init>,       // tp_init
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
// Helper functions
//------------------------------------------------------------------------------

using PyDaytimeDefault = PyDaytime<cron::Daytime>;

inline ref<Object>
make_daytime(
  cron::Daytick const daytick,
  Object* type=(Object*) &PyDaytimeDefault::type_)
{
  // Special case fast path for the default daytime type.
  if (type == (Object*) &PyDaytimeDefault::type_)
    return PyDaytimeDefault::create(
      PyDaytimeDefault::Daytime::from_daytick(daytick));
  else
    return type->CallMethodObjArgs(
      "from_daytick", Long::FromUnsignedLong(daytick));
}


template<typename DAYTIME>
inline DAYTIME
parts_to_daytime(
  Sequence* const parts)
{
  // Interpret a two- or three-element sequence as parts.
  long   const hour     = parts->GetItem(0)->long_value();
  long   const minute   = parts->GetItem(1)->long_value();
  double const second
    = parts->Length() > 2 ? parts->GetItem(2)->double_value() : 0;
  return DAYTIME::from_parts(hour, minute, second);
}


template<typename DAYTIME>
inline optional<DAYTIME>
maybe_daytime(
  Object* const obj)
{
  if (PyDaytime<DAYTIME>::Check(obj))
    // Exact wrapped type.
    return static_cast<PyDaytime<DAYTIME>*>(obj)->daytime_;

  // Try for a 'datetime.time' instance.
  if (PyDateTimeAPI == nullptr)
    PyDateTime_IMPORT;
  if (PyTime_Check(obj))
    return DAYTIME(
      PyDateTime_TIME_GET_HOUR(obj),
      PyDateTime_TIME_GET_MINUTE(obj),
      PyDateTime_TIME_GET_SECOND(obj)
      + 1e-6 * PyDateTime_TIME_GET_MICROSECOND(obj));

  // Try for a daytime type that has a 'daytick' attribute.
  auto daytick = obj->GetAttrString("daytick", false);
  if (daytick != nullptr)
    return DAYTIME::from_daytick(daytick->long_value());

  // No type match.
  return {};
}


template<typename DAYTIME>
inline DAYTIME
convert_to_daytime(
  Object* const obj)
{
  if (obj == None)
    // Use the default value.
    return DAYTIME{};

  auto opt = maybe_daytime<DAYTIME>(obj);
  if (opt)
    return *opt;

  if (Sequence::Check(obj)) 
    return parts_to_daytime<DAYTIME>(static_cast<Sequence*>(obj));

  auto const double_opt = obj->maybe_double_value();
  if (double_opt) 
    // Interpret as SSM.
    return DAYTIME::from_ssm(*double_opt);
      
  // FIXME: Parse strings.

  // Failed to convert.
  throw py::TypeError("can't convert to daytime: "s + *obj->Repr());
}


//------------------------------------------------------------------------------

}  // namespace aslib

