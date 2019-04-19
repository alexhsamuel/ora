#pragma once
#pragma GCC diagnostic ignored "-Wparentheses"

#include <cmath>
#include <experimental/optional>
#include <iostream>
#include <unordered_map>

#include <Python.h>
#include <datetime.h>

#include "np.hh"
#include "ora.hh"
#include "py.hh"
#include "types.hh"

namespace ora {
namespace py {

using namespace std::literals;

using std::experimental::optional;
using std::make_unique;
using std::string;
using std::unique_ptr;

//------------------------------------------------------------------------------
// Declarations
//------------------------------------------------------------------------------

/**
 * Attempts to convert various kinds of Python daytime object to 'DAYTIME'.
 *
 * If 'obj' is a daytime object of some kind, returns the equivalent daytime;
 * otherwise a null option.  The following daytime objects are recognized:
 *
 *  - PyDaytimeTemplate instances
 *  - 'datetime.time` instances
 */
template<class DAYTIME> optional<DAYTIME> maybe_daytime(Object*);

/**
 * Converts various kinds of Python objects to Daytime.
 *
 * If 'obj' can be converted unambiguously to a daytime, returns it; otherwise
 * rasies a Python exception.
 */
template<class DAYTIME> DAYTIME convert_to_daytime(Object*);

/**
 * Helper for converting a 2- or 3-eleme4nt sequence of daytime parts.
 */
template<class DAYTIME> inline DAYTIME parts_to_daytime(Sequence*);

//------------------------------------------------------------------------------
// Virtual API
//------------------------------------------------------------------------------

/*
 * Provides an PI with dynamic dispatch to PyDaytime objects.
 */
class PyDaytimeAPI
{
public:

  virtual ~PyDaytimeAPI() {}

  /*
   * Registers a virtual API for a Python type.
   */
  static void add(PyTypeObject* const type, std::unique_ptr<PyDaytimeAPI>&& api)
    { apis_.emplace(type, std::move(api)); }

  /*
   * Returns the API for a Python object, or nullptr if it isn't a PyDate.
   */
  static PyDaytimeAPI const*
  get(
    PyTypeObject* const type)
  {
    auto api = apis_.find(type);
    return api == apis_.end() ? nullptr : api->second.get();
  }

  static PyDaytimeAPI const* get(PyObject* const obj)
    { return get(obj->ob_type); }

  // API methods.
  virtual ora::Daytick              get_daytick(Object* daytime) const = 0;
  virtual ref<Object>               from_daytick(ora::Daytick) const = 0;
  virtual ref<Object>               from_hms(HmsDaytime) const = 0;
  virtual bool                      is_invalid(Object* daytime) const = 0;
  virtual bool                      is_missing(Object* daytime) const = 0;

private:

  static std::unordered_map<PyTypeObject*, std::unique_ptr<PyDaytimeAPI>> apis_;

};

//------------------------------------------------------------------------------
// Docstrings
//------------------------------------------------------------------------------

namespace docstring {
namespace pydaytime {

#include "py_daytime.docstrings.hh.inc"

}  // namespace pydaytime
}  // namespace docstring

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
template<class DAYTIME>
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

  class API
  : public PyDaytimeAPI
  {
  public:

    virtual ora::Daytick get_daytick(Object* const daytime) const
      { return ((PyDaytime*) daytime)->daytime_.get_daytick(); }
    virtual ref<Object> from_daytick(ora::Daytick const daytick) const
      { return PyDaytime::create(ora::daytime::from_daytick<Daytime>(daytick)); }
    virtual ref<Object> from_hms(HmsDaytime const parts) const
      { return PyDaytime::create(ora::daytime::from_hms(parts)); }
    virtual bool is_invalid(Object* const daytime) const
      { return ((PyDaytime*) daytime)->daytime_.is_invalid(); }
    virtual bool is_missing(Object* const daytime) const
      { return ((PyDaytime*) daytime)->daytime_.is_missing(); }

  };

  static void tp_init(PyDaytime* self, Tuple* args, Dict* kw_args);
  static void tp_dealloc(PyDaytime* self);
  static ref<Unicode> tp_repr(PyDaytime* self);
  static Py_hash_t    tp_hash(PyDaytime* self);
  static ref<Unicode> tp_str(PyDaytime* self);
  static ref<Object>  tp_richcompare(PyDaytime* self, Object* other, int comparison);

  // Number methods.
  static ref<Object> nb_add     (PyDaytime* self, Object* other, bool right);
  static ref<Object> nb_subtract(PyDaytime* self, Object* other, bool right);
  static ref<Object> nb_int     (PyDaytime* self);
  static ref<Object> nb_float   (PyDaytime* self);
  static PyNumberMethods tp_as_number_;

  // Methods.
  static ref<Object> method___format__          (PyDaytime*, Tuple*, Dict*);
  static ref<Object> method_from_daytick        (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_hms            (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static ref<Object> method_from_ssm            (PyTypeObject* type, Tuple* args, Dict* kw_args);
  static Methods<PyDaytime> tp_methods_;

  // Getsets.
  static ref<Object> get_daytick                (PyDaytime* self, void*);
  static ref<Object> get_hms                    (PyDaytime* self, void*);
  static ref<Object> get_hour                   (PyDaytime* self, void*);
  static ref<Object> get_invalid                (PyDaytime* self, void*);
  static ref<Object> get_minute                 (PyDaytime* self, void*);
  static ref<Object> get_missing                (PyDaytime* self, void*);
  static ref<Object> get_offset                 (PyDaytime* self, void*);
  static ref<Object> get_second                 (PyDaytime* self, void*);
  static ref<Object> get_ssm                    (PyDaytime* self, void*);
  static ref<Object> get_std                    (PyDaytime* self, void*);
  static ref<Object> get_valid                  (PyDaytime* self, void*);
  static GetSets<PyDaytime> tp_getsets_;

  static int precision_;
  /** Date format used to generate the repr and str.  */
  static unique_ptr<ora::daytime::DaytimeFormat> repr_format_;

  static Type build_type(string const& type_name);

};


template<class DAYTIME>
void
PyDaytime<DAYTIME>::add_to(
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
  PyDaytimeAPI::add(&type_, std::make_unique<API>());

  // Choose precision for seconds that captures actual precision of the daytime
  // class (up to 1 fs).
  precision_ = std::min((size_t) ceil(log10(DAYTIME::DENOMINATOR)), 15ul);

  // Build the repr format.
  repr_format_ = make_unique<ora::daytime::DaytimeFormat>(
    name + "(%0H, %0M, " + std::string("%02")
    + (precision_ > 0 ? "." + std::to_string(precision_) : "") + "S)",
    name + ".INVALID",
    name + ".MISSING");

  // Add in static data members.
  auto const dict = (Dict*) type_.tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("DENOMINATOR" , Long::from(DAYTIME::DENOMINATOR));
  dict->SetItemString("INVALID"     , create(Daytime::INVALID));
  dict->SetItemString("MAX"         , create(Daytime::MAX));
  dict->SetItemString("MIDNIGHT"    , create(Daytime::MIDNIGHT));
  dict->SetItemString("MIN"         , create(Daytime::MIN));
  dict->SetItemString("MISSING"     , create(Daytime::MISSING));
  dict->SetItemString("RESOLUTION"  , Float::FromDouble(DAYTIME::RESOLUTION));

  // Add the type to the module.
  module.add(&type_);
}


template<class DAYTIME>
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


template<class DAYTIME>
Type
PyDaytime<DAYTIME>::type_;


template<class DAYTIME>
bool
PyDaytime<DAYTIME>::Check(
  PyObject* const other)
{
  return static_cast<Object*>(other)->IsInstance((PyObject*) &type_);
}


//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

template<class DAYTIME>
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


template<class DAYTIME>
void
PyDaytime<DAYTIME>::tp_dealloc(
  PyDaytime* const self)
{
  self->daytime_.~DaytimeTemplate();
  self->ob_type->tp_free(self);
}


template<class DAYTIME>
ref<Unicode>
PyDaytime<DAYTIME>::tp_repr(
  PyDaytime* const self)
{
  return Unicode::from((*repr_format_)(self->daytime_));
}


template<class DAYTIME>
Py_hash_t
PyDaytime<DAYTIME>::tp_hash(
  PyDaytime* const self)
{
  return 
      self->daytime_.is_invalid() ? std::numeric_limits<Py_hash_t>::max()
    : self->daytime_.is_missing() ? std::numeric_limits<Py_hash_t>::max() - 1
    : self->daytime_.get_offset();
}


template<class DAYTIME>
ref<Unicode>
PyDaytime<DAYTIME>::tp_str(
  PyDaytime* const self)
{
  if (self->daytime_.is_invalid())
    return Unicode::from("INVALID");
  else if (self->daytime_.is_missing())
    return Unicode::from("MISSING");
  else {
    auto const hms = daytime::get_hms(self->daytime_);
    StringBuilder sb;
    daytime::format_iso_daytime(sb, hms, precision_);
    return Unicode::FromStringAndSize((char const*) sb, sb.length());
  }
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::tp_richcompare(
  PyDaytime* const self,
  Object* const other,
  int const comparison)
{
  auto const opt = maybe_daytime<DAYTIME>(other);
  if (!opt)
    return not_implemented_ref();
  return richcmp(self->daytime_, *opt, comparison);
}


//------------------------------------------------------------------------------
// Number methods
//------------------------------------------------------------------------------

template<class DAYTIME>
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


template<class DAYTIME>
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


template<class DAYTIME>
inline ref<Object>
PyDaytime<DAYTIME>::nb_int(
  PyDaytime* const self)
{
  throw TypeError("int() argument cannot be a daytime");
}


template<class DAYTIME>
inline ref<Object>
PyDaytime<DAYTIME>::nb_float(
  PyDaytime* const self)
{
  throw TypeError("float() argument cannot be a daytime");
}


template<class DAYTIME>
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
  (unaryfunc)   wrap<PyDaytime, nb_int>,        // nb_int
  (void*)       nullptr,                        // nb_reserved
  (unaryfunc)   wrap<PyDaytime, nb_float>,      // nb_float
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

template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::method___format__(
  PyDaytime* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  if (args->GetLength() != 1 || kw_args != nullptr)
    throw TypeError("__format__() takes one argument");
  auto const fmt = args->GetItem(0)->Str()->as_utf8();

  if (*fmt == '\0')
    return tp_str(self);
  else
    return Unicode::from(ora::DaytimeFormat(fmt)(self->daytime_));
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::method_from_daytick(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"daytick", nullptr};
  ora::Daytick daytick;
  static_assert(
    sizeof(ora::Daytick) == sizeof(long), "daytick is not an long");
  Arg::ParseTupleAndKeywords(args, kw_args, "k", arg_names, &daytick);

  return create(ora::daytime::from_daytick<Daytime>(daytick), type);
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::method_from_hms(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  if (kw_args != nullptr)
    throw TypeError("from_hms() takes no keyword arguments");

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
    throw TypeError("from_hms() takes one or three arguments");

  long   const hour   = parts->GetItem(0)->long_value();
  long   const minute = parts->GetItem(1)->long_value();
  double const second
    = parts->Length() == 3 ? parts->GetItem(2)->double_value() : 0;
  return create(ora::daytime::from_hms<Daytime>(hour, minute, second), type);
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::method_from_ssm(
  PyTypeObject* const type,
  Tuple* const args,
  Dict* const kw_args)
{
  static char const* const arg_names[] = {"ssm", nullptr};
  ora::Ssm ssm;
  Arg::ParseTupleAndKeywords(args, kw_args, "d", arg_names, &ssm);

  return create(ora::daytime::from_ssm<Daytime>(ssm), type);
}


template<class DAYTIME>
Methods<PyDaytime<DAYTIME>>
PyDaytime<DAYTIME>::tp_methods_
  = Methods<PyDaytime>()
    .template add<method___format__>                ("__format__")
    .template add_class<method_from_daytick>        ("from_daytick",    docstring::pydaytime::from_daytick)
    .template add_class<method_from_hms>            ("from_hms",        docstring::pydaytime::from_hms)
    .template add_class<method_from_ssm>            ("from_ssm",        docstring::pydaytime::from_ssm)
  ;

//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_daytick(
  PyDaytime* self,
  void* /* closure */)
{
  
  return Long::FromUnsignedLong(self->daytime_.get_daytick());
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_hms(
  PyDaytime* self,
  void* /* closure */)
{
  return make_hms_daytime(ora::daytime::get_hms(self->daytime_));
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_hour(
  PyDaytime* self,
  void* /* closure */)
{
  return Long::FromLong(ora::daytime::get_hour(self->daytime_));
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_invalid(
  PyDaytime* self,
  void* /* closure */)
{
  return Bool::from(self->daytime_.is_invalid());
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_minute(
  PyDaytime* self,
  void* /* closure */)
{
  return Long::FromLong(ora::daytime::get_minute(self->daytime_));
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_missing(
  PyDaytime* self,
  void* /* closure */)
{
  return Bool::from(self->daytime_.is_missing());
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_offset(
  PyDaytime* self,
  void* /* closure */)
{
  return Long::FromLong(self->daytime_.get_offset());
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_second(
  PyDaytime* self,
  void* /* closure */)
{
  return Float::FromDouble(ora::daytime::get_second(self->daytime_));
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_ssm(
  PyDaytime* self,
  void* /* closure */)
{
  return Float::FromDouble(ora::daytime::get_ssm(self->daytime_));
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_std(
  PyDaytime* self,
  void* /* closure */)
{
  using Offset = typename DAYTIME::Offset;

  if (!self->daytime_.is_valid())
    throw py::ValueError("daytime not valid");

  if (PyDateTimeAPI == nullptr)
    PyDateTime_IMPORT;
  
  auto const usec = ora::UsecDaytime(self->daytime_).get_offset();
  return ref<Object>::take(PyTime_FromTime(
    usec / (Offset) 3600000000, 
    usec % (Offset) 3600000000 / 60000000,
    usec % (Offset)   60000000 /  1000000,
    usec % (Offset)    1000000));
}


template<class DAYTIME>
ref<Object>
PyDaytime<DAYTIME>::get_valid(
  PyDaytime* self,
  void* /* closure */)
{
  return Bool::from(self->daytime_.is_valid());
}


template<class DAYTIME>
GetSets<PyDaytime<DAYTIME>>
PyDaytime<DAYTIME>::tp_getsets_ 
  = GetSets<PyDaytime>()
    .template add_get<get_daytick>              ("daytick"  , docstring::pydaytime::daytick)
    .template add_get<get_hms>                  ("hms"      , docstring::pydaytime::hms)
    .template add_get<get_hour>                 ("hour"     , docstring::pydaytime::hour)
    .template add_get<get_invalid>              ("invalid"  , docstring::pydaytime::invalid)
    .template add_get<get_minute>               ("minute"   , docstring::pydaytime::minute)
    .template add_get<get_missing>              ("missing"  , docstring::pydaytime::missing)
    .template add_get<get_offset>               ("offset"   , docstring::pydaytime::offset)
    .template add_get<get_second>               ("second"   , docstring::pydaytime::second)
    .template add_get<get_ssm>                  ("ssm"      , docstring::pydaytime::ssm)
    .template add_get<get_std>                  ("std"      , docstring::pydaytime::std)
    .template add_get<get_valid>                ("valid"    , docstring::pydaytime::valid)
  ;


//------------------------------------------------------------------------------
// Other members
//------------------------------------------------------------------------------

template<class DAYTIME>
int
PyDaytime<DAYTIME>::precision_;

template<class DAYTIME>
unique_ptr<ora::daytime::DaytimeFormat>
PyDaytime<DAYTIME>::repr_format_;

//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

template<class DAYTIME>
Type
PyDaytime<DAYTIME>::build_type(
  string const& type_name)
{
  // FIXME: Factor out somewhere.
  auto const doc_len    = strlen(docstring::pydaytime::type) + 64;
  auto const doc        = new char[doc_len];
  auto const dot        = type_name.find_last_of('.');
  auto unqualified_name = 
    dot == string::npos ? type_name : type_name.substr(dot + 1);
  snprintf(
    doc, doc_len, docstring::pydaytime::type,
    unqualified_name.c_str(),
    DAYTIME::RESOLUTION);

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
    (hashfunc)            wrap<PyDaytime, tp_hash>,       // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            wrap<PyDaytime, tp_str>,        // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         doc,                            // tp_doc
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

using PyDaytimeDefault = PyDaytime<ora::daytime::Daytime>;

inline ref<Object>
make_daytime(
  ora::Daytick const daytick,
  PyTypeObject* type=&PyDaytimeDefault::type_)
{
  // Special case fast path for the default daytime type.
  if (type == &PyDaytimeDefault::type_)
    return PyDaytimeDefault::create(
      ora::daytime::from_daytick<PyDaytimeDefault::Daytime>(daytick));

  auto const api = PyDaytimeAPI::get(type);
  if (api == nullptr)
    throw TypeError("not a daytime type: "s + *(((Object*) type)->Repr()));
  else
    return api->from_daytick(daytick);
}


template<class DAYTIME>
inline DAYTIME
parts_to_daytime(
  Sequence* const parts)
{
  // Interpret a two- or three-element sequence as parts.
  long   const hour     = parts->GetItem(0)->long_value();
  long   const minute   = parts->GetItem(1)->long_value();
  double const second
    = parts->Length() > 2 ? parts->GetItem(2)->double_value() : 0;
  return ora::daytime::from_hms<DAYTIME>(hour, minute, second);
}


template<class DAYTIME>
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
    return ora::daytime::from_hms<DAYTIME>(
      PyDateTime_TIME_GET_HOUR(obj),
      PyDateTime_TIME_GET_MINUTE(obj),
        PyDateTime_TIME_GET_SECOND(obj)
      + 1e-6 * PyDateTime_TIME_GET_MICROSECOND(obj));

  // Try for a daytime type that has a 'daytick' attribute.
  auto daytick = obj->GetAttrString("daytick", false);
  if (daytick != nullptr)
    return ora::daytime::from_daytick<DAYTIME>(daytick->long_value());

  // No type match.
  return {};
}


template<class DAYTIME>
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

  if (Unicode::Check(obj)) {
    auto const str = static_cast<Unicode*>(obj)->as_utf8_string();
    try {
      return ora::daytime::from_iso_daytime<DAYTIME>(str);
    }
    catch (ora::DaytimeError const&) {
      throw py::ValueError("can't parse as daytime: '"s + str + "'");
    }
  }

  if (Sequence::Check(obj)) 
    return parts_to_daytime<DAYTIME>(static_cast<Sequence*>(obj));

  auto const double_opt = obj->maybe_double_value();
  if (double_opt) 
    // Interpret as SSM.
    return ora::daytime::from_ssm<DAYTIME>(*double_opt);
      
  // Failed to convert.
  throw py::TypeError("can't convert to daytime: "s + *obj->Repr());
}


//------------------------------------------------------------------------------

#ifdef __clang__
// Use explicit instantiation for the main instances.
// FIXME: GCC 5.2.1 generates PyDaytime<>::type_ in BSS, which breaks linking.
extern template class PyDaytime<ora::daytime::Daytime>;
extern template class PyDaytime<ora::daytime::Daytime32>;
extern template class PyDaytime<ora::daytime::UsecDaytime>;
#endif

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

