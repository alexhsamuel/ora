#pragma once

#include <cassert>
#include <experimental/optional>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include <Python.h>

//------------------------------------------------------------------------------

namespace ora {
namespace py {

using std::experimental::optional;

class Float;
class Long;
class Object;
class Tuple;
class Type;
class Unicode;

//------------------------------------------------------------------------------

class Exception
{
public:

  Exception() {}
  
  Exception(PyObject* exception, char const* message)
  {
    PyErr_SetString(exception, message);
  }

  template<class A>
  Exception(PyObject* exception, A&& message)
  {
    PyErr_SetString(exception, std::string(std::forward<A>(message)).c_str());
  }

  /**
   * Clears up the Python exception state.  
   */
  static void Clear() { PyErr_Clear(); }

};


/**
 * Template wrapper for a specific Python exception type.
 */
template<PyObject** EXC>
class ExceptionWrapper
  : public Exception
{
public:

  template<class A>
  ExceptionWrapper(A&& message)
    : Exception(*EXC, std::forward<A>(message))
  {}

};


using ArithmeticError       = ExceptionWrapper<&PyExc_ArithmeticError>;
using AttributeError        = ExceptionWrapper<&PyExc_AttributeError>;
using EnvironmentError      = ExceptionWrapper<&PyExc_EnvironmentError>;
using FileExistsError       = ExceptionWrapper<&PyExc_FileExistsError>;
using FileNotFoundError     = ExceptionWrapper<&PyExc_FileNotFoundError>;
using IOError               = ExceptionWrapper<&PyExc_IOError>;
using ImportError           = ExceptionWrapper<&PyExc_ImportError>;
using IndexError            = ExceptionWrapper<&PyExc_IndexError>;
using InterruptedError      = ExceptionWrapper<&PyExc_InterruptedError>;
using IsADirectoryError     = ExceptionWrapper<&PyExc_IsADirectoryError>;
using KeyError              = ExceptionWrapper<&PyExc_KeyError>;
using LookupError           = ExceptionWrapper<&PyExc_LookupError>;
using NameError             = ExceptionWrapper<&PyExc_NameError>;
using NotADirectoryError    = ExceptionWrapper<&PyExc_NotADirectoryError>;
using NotImplementedError   = ExceptionWrapper<&PyExc_NotImplementedError>;
using OverflowError         = ExceptionWrapper<&PyExc_OverflowError>;
using PermissionError       = ExceptionWrapper<&PyExc_PermissionError>;
using ReferenceError        = ExceptionWrapper<&PyExc_ReferenceError>;
using RuntimeError          = ExceptionWrapper<&PyExc_RuntimeError>;
using StopIteration         = ExceptionWrapper<&PyExc_StopIteration>;
using SystemExit            = ExceptionWrapper<&PyExc_SystemExit>;
using TimeoutError          = ExceptionWrapper<&PyExc_TimeoutError>;
using TypeError             = ExceptionWrapper<&PyExc_TypeError>;
using ValueError            = ExceptionWrapper<&PyExc_ValueError>;
using ZeroDivisionError     = ExceptionWrapper<&PyExc_ZeroDivisionError>;


/**
 * Raises 'Exception' if value is not zero.
 */
inline void check_zero(int value)
{
  assert(value == 0 || value == -1);
  if (value != 0)
    throw Exception();
}


/*
 * Raises 'Exception' if value is not one.
 */
inline void
check_one(
  int const value)
{
  assert(value == 0 || value == 1);
  if (value != 1)
    throw Exception();
}


/**
 * Raises 'Exception' if 'value' is -1; returns it otherwise.
 */
template<class TYPE>
inline TYPE check_not_minus_one(TYPE value)
{
  if (value == -1)
    throw Exception();
  else
    return value;
}


/**
 * Raises 'Exception' if value is not true.
 */
inline void check_true(int value)
{
  if (value == 0)
    throw Exception();
}


/**
 * Raises 'Exception' if 'value' is null; otherwise, returns it.
 */
inline Object* check_not_null(PyObject* obj)
{
  if (obj == nullptr)
    throw Exception();
  else
    return (Object*) obj;
}


inline Type* check_not_null(PyTypeObject* type)
{
  if (type == nullptr)
    throw Exception();
  else
    return (Type*) type;
}


//------------------------------------------------------------------------------

template<class T>
inline T* 
cast(PyObject* obj)
{
  assert(T::Check(obj));  // FIXME: TypeError?
  return static_cast<T*>(obj);
}


inline PyObject* incref(PyObject* obj)
{
  Py_INCREF(obj);
  return obj;
}


inline PyTypeObject* 
incref(
  PyTypeObject* obj)
{
  Py_INCREF(obj);
  return obj;
}


inline PyObject* decref(PyObject* obj)
{
  Py_DECREF(obj);
  return obj;
}


//------------------------------------------------------------------------------

/**
 * Type-generic base class for references.
 *
 * An instance of this class owns a reference to a Python object.
 */
class baseref
{
public:

  ~baseref() 
  {
    clear();
  }

  Object* release()
  {
    auto obj = obj_;
    obj_ = nullptr;
    return obj;
  }

  void clear();

protected:

  baseref(Object* obj) : obj_(obj) {}

  Object* obj_;

};


// FIXME: ref<> does not work with Type and derived objects, since it's not
// a subtype of Object.  This would be hard to do because Object derives from
// PyObject, which declares data attributes, the same as PyTypeObject.
//
// One way around this would be to separate the mixin Object, like PyObject
// with extra helpers, from a BaseObject type that derives from the concrete
// PyObject.

template<class T>
class ref
  : public baseref
{
public:

  /**
   * Takes an existing reference.
   *
   * Call this method on an object pointer that comes with an assumed reference,
   * such as the return value of an API call that returns ownership.
   */
  // FIXME: Maybe do an unchecked cast here, since C API functions do a check?
  static ref<T> take(PyObject* obj)
    { return ref(cast<T>(obj)); }

  /**
   * Creates a new reference.
   */
  static ref<T> of(ref<T> obj_ref)
    { return of(obj_ref.obj_); }

  /**
   * Creates a new reference.
   */
  static ref<T> of(T* obj)
    { incref(obj); return ref{obj}; }

  /**
   * Creates a new reference, casting.
   */
  static ref<T> of(PyObject* obj)
    { return of(cast<T>(obj)); }

  /** 
   * Default ctor: null reference.  
   */
  ref()
    : baseref(nullptr) {}

  /** 
   * Move ctor.  
   */
  ref(ref<T>&& ref)
    : baseref(ref.release()) {}

  /** 
   * Move ctor from another ref type.  
   */
  template<class U>
  ref(ref<U>&& ref)
    : baseref(ref.release()) {}

  /**
   * Returns a new (additional) reference.
   */
  ref inc() const
    { return ref::of(obj_); }

  void operator=(ref<T>&& ref)
    { clear(); obj_ = ref.release(); }

  bool operator==(T* const ptr) const
    { return obj_ == ptr; }

  bool operator!=(T* const ptr) const
    { return obj_ != ptr; }

  operator T&() const
    { return *(T*) obj_; }

  operator T*() const
    { return (T*) obj_; }

  T* operator->() const
    { return (T*) obj_; }

  T* release()
    { return (T*) baseref::release(); }

private:

  ref(T* obj)
    : baseref(obj) 
  {}

};


inline ref<Object> none_ref()
{
  return ref<Object>::of(Py_None);
}


inline ref<Object> not_implemented_ref()
{
  return ref<Object>::of(Py_NotImplemented);
}


template<class TYPE>
inline ref<TYPE>
take_not_null(
  PyObject* obj)
{
  if (obj == nullptr)
    throw Exception();
  else
    return ref<TYPE>::take(obj);
}


//------------------------------------------------------------------------------

template<class T>
inline ref<T>
cast(ref<Object>&& obj)
{
  return ref<T>::take(obj.release());
}


//==============================================================================

class Object
  : public PyObject
{
public:

  ref<Object> CallMethodObjArgs(char const* name, bool check=true);
  // FIXME: Hacky.
  ref<Object> CallMethodObjArgs(char const* name, PyObject* arg0, bool check=true);
  ref<Object> CallMethodObjArgs(char const* name, PyObject* arg0, PyObject* arg1, bool check=true);
  ref<Object> CallObject(Tuple* args);
  static bool Check(PyObject* obj)
    { return true; }
  ref<Object> GetAttrString(char const* const name, bool check=true);
  bool IsInstance(PyObject* type)
    { return (bool) PyObject_IsInstance(this, type); }
  bool IsInstance(PyTypeObject* type)
    { return IsInstance((PyObject*) type); }
  auto Length()
    { return PyObject_Length(this); }
  auto Repr()
    { return ref<Unicode>::take(PyObject_Repr(this)); }
  auto SetAttrString(char const* name, PyObject* obj)
    { check_not_minus_one(PyObject_SetAttrString(this, name, obj)); }
  auto Str()
    { return ref<Unicode>::take(PyObject_Str(this)); }

  optional<ref<Object>> maybe_get_attr(std::string const& name);

  ref<py::Long> Long(bool check=true);
  long long_value();
  unsigned long unsigned_long_value();
  /** If the object can be converted to a long, returns its value. */
  optional<long> maybe_long_value();

  ref<py::Float> Float();
  double double_value();
  /** If the object can be converted to a long, returns its value. */
  optional<double> maybe_double_value();


};


inline ref<Object>
Object::CallMethodObjArgs(
  char const* name,
  bool check)
{
  ref<Object> method = GetAttrString(name, check);
  if (!check && method == nullptr) {
    Exception::Clear();
    return ref<Object>::take(nullptr);
  }
  auto result = PyObject_CallFunctionObjArgs(method, nullptr);
  if (check)
    check_not_null(result);
  // FIXME: Clumsy.
  else if (result == nullptr)
    Exception::Clear();
  return ref<Object>::take(result);
}


inline ref<Object>
Object::CallMethodObjArgs(
  char const* name,
  PyObject* arg0,
  bool check)
{
  ref<Object> method = GetAttrString(name, check);
  if (!check && method == nullptr) {
    Exception::Clear();
    return ref<Object>::take(nullptr);
  }
  auto result = PyObject_CallFunctionObjArgs(method, arg0, nullptr);
  if (check)
    check_not_null(result);
  // FIXME: Clumsy.
  else if (result == nullptr)
    Exception::Clear();
  return ref<Object>::take(result);
}


inline ref<Object>
Object::CallMethodObjArgs(
  char const* name,
  PyObject* arg0,
  PyObject* arg1,
  bool check)
{
  ref<Object> method = GetAttrString(name, check);
  if (!check && method == nullptr) {
    Exception::Clear();
    return ref<Object>::take(nullptr);
  }
  auto result = PyObject_CallFunctionObjArgs(method, arg0, arg1, nullptr);
  if (check)
    check_not_null(result);
  // FIXME: Clumsy.
  else if (result == nullptr)
    Exception::Clear();
  return ref<Object>::take(result);
}


inline ref<Object>
Object::GetAttrString(
  char const* const name, 
  bool const check)
{ 
  auto result = PyObject_GetAttrString(this, name);
  if (check)
    result = check_not_null(result);
  // FIXME: Clumsy.
  else if (result == nullptr)
    Exception::Clear();
  return ref<Object>::take(result);
}


inline optional<ref<Object>>
Object::maybe_get_attr(
  std::string const& name)
{ 
  auto result = PyObject_GetAttrString(this, name.c_str());
  // FIXME: Encapsulate this pattern.
  if (result == nullptr) {
    Exception::Clear();
    return {};
  }
  else
    return ref<Object>::take(result);
}


template<class T>
inline std::ostream& operator<<(std::ostream& os, ref<T>& ref)
{
  os << ref->Str()->as_utf8();
  return os;
}


extern ref<Object> const
None;


//------------------------------------------------------------------------------

class Dict
  : public Object
{
public:

  static bool Check(PyObject* const obj)
    { return PyDict_Check(obj); }

  Object* GetItemString(char const* const key)
  { 
    Object* const value = (Object*) PyDict_GetItemString(this, key);
    if (value == nullptr)
      throw KeyError(key);
    else
      return value;
  }

  // FIXME: Add an rvalue variant that takes the ref?
  void SetItemString(char const* const key, PyObject* const value)
    { check_zero(PyDict_SetItemString(this, key, value)); }

  Py_ssize_t Size()
    { return check_not_minus_one(PyDict_Size(this)); }

};


//------------------------------------------------------------------------------

class Bool
  : public Object
{
public:

  static ref<Bool> const TRUE;
  static ref<Bool> const FALSE;

  static bool Check(PyObject* obj)
    { return PyBool_Check(obj); }
  static auto from(bool value)
    { return ref<Bool>::of(value ? Py_True : Py_False); }

  operator bool()
    { return this == Py_True; }

};


//------------------------------------------------------------------------------

class Number
  : public Object
{
public:

  auto Lshift(PyObject* rhs)
    { return take_not_null<Number>(PyNumber_Lshift(this, rhs)); }
  auto Or(PyObject* rhs)
    { return take_not_null<Number>(PyNumber_Or(this, rhs)); }

};

//------------------------------------------------------------------------------

class Long
  : public Number
{
public:

  static bool Check(PyObject* obj)
    { return PyLong_Check(obj); }
  static auto FromLong(long val)
    { return ref<Long>::take(PyLong_FromLong(val)); }
  static auto FromUnsignedLong(unsigned long val)
    { return ref<Long>::take(PyLong_FromUnsignedLong(val)); }

  static ref<Long> from(int const val)
    { return FromLong(val); }
  static ref<Long> from(unsigned int const val)
    { return FromUnsignedLong(val); }
  static ref<Long> from(long const val)
    { return FromLong(val); }
  static ref<Long> from(unsigned long const val)
    { return FromUnsignedLong(val); }
  static ref<Long> from(long long const val)
    { return FromLong(val); }
  static ref<Long> from(unsigned long long const val)
    { return FromUnsignedLong(val); }
  static ref<Long> from(__int128 const val);
  static ref<Long> from(unsigned __int128 const val);

  operator long()
    { return PyLong_AsLong(this); }
  operator unsigned long()
    { return PyLong_AsUnsignedLong(this); }

  explicit operator __int128();
  explicit operator unsigned __int128();

};


inline ref<Long> 
Long::from(__int128 val)
{ 
  return take_not_null<Long>(
    _PyLong_FromByteArray((unsigned char const*) &val, sizeof(val), 1, 1));
}


inline ref<Long> 
Long::from(unsigned __int128 val)
{ 
  return take_not_null<Long>(
    _PyLong_FromByteArray((unsigned char const*) &val, sizeof(val), 1, 0));
}


inline 
Long::operator __int128()
{
  __int128 val = 0;
  check_not_minus_one(_PyLong_AsByteArray(
    (PyLongObject*) this, (unsigned char*) &val, sizeof(val), 1, 1));
  return val;
}


inline
Long::operator unsigned __int128()
{
  unsigned __int128 val;
  check_not_minus_one(_PyLong_AsByteArray(
    (PyLongObject*) this, (unsigned char*) &val, sizeof(val), 1, 0));
  return val;
}


//------------------------------------------------------------------------------

class Float
  : public Object
{
public:

  static bool Check(PyObject* obj)
    { return PyFloat_Check(obj); }
  static auto FromDouble(double val)
    { return ref<Float>::take(PyFloat_FromDouble(val)); }
  static auto from(double const val)
    { return FromDouble(val); }

  operator double()
    { return PyFloat_AsDouble(this); }

};


//------------------------------------------------------------------------------

class Module
  : public Object
{
public:

  static bool Check(PyObject* obj)
    { return PyModule_Check(obj); }
  static auto Create(PyModuleDef* def)
    { return ref<Module>::take(PyModule_Create(def)); }
  static ref<Module> ImportModule(char const* const name)
    { return take_not_null<Module>(PyImport_ImportModule(name)); }
  static ref<Module> New(char const* name)
    { return take_not_null<Module>(PyModule_New(name)); }

  void AddFunctions(PyMethodDef* functions) 
    { check_zero(PyModule_AddFunctions(this, functions)); }
  void AddObject(char const* name, PyObject* val)
    { check_zero(PyModule_AddObject(this, name, incref(val))); }
  char const* GetName()
    { return PyModule_GetName(this); }

  void add(PyTypeObject* type)
  {
    // Make sure the qualified name of the type includes this module's name.
    std::string const qualname = type->tp_name;
    std::string const mod_name = PyModule_GetName(this);
    auto dot = qualname.find_last_of('.');
    assert(dot != std::string::npos);
    assert(qualname.compare(0, dot, mod_name) == 1);
    // Add it, under its unqualified name.
    AddObject(qualname.substr(dot + 1).c_str(), (PyObject*) type);
  }

};


//------------------------------------------------------------------------------

class Sequence
  : public Object
{
public:

  static bool Check(PyObject* obj)
    { return PySequence_Check(obj); }

  Object* GetItem(Py_ssize_t index)
    { return check_not_null(PySequence_GetItem(this, index)); }

  Py_ssize_t Length()
    { return PySequence_Length(this); }

};


//------------------------------------------------------------------------------

class Tuple
  : public Sequence
{
public:

  static bool Check(PyObject* obj)
    { return PyTuple_Check(obj); }

  static auto New(Py_ssize_t len)
    { return ref<Tuple>::take(PyTuple_New(len)); }

  void initialize(Py_ssize_t index, baseref&& ref)
    { PyTuple_SET_ITEM(this, index, ref.release()); }

  Object* GetItem(Py_ssize_t index) 
    { return check_not_null(PyTuple_GET_ITEM(this, index)); }

  Py_ssize_t GetLength() 
    { return PyTuple_GET_SIZE(this); }

  // FIXME: Remove?
  static auto from(std::initializer_list<PyObject*> items) 
  {
    auto len = items.size();
    auto tuple = New(len);
    Py_ssize_t index = 0;
    for (auto item : items) 
      PyTuple_SET_ITEM((PyObject*) tuple, index++, item);
    return tuple;
  }

private:

  /**
   * Recursive template for building fixed-sized tuples.
   */
  template<Py_ssize_t LEN> class Builder;

public:

  static Builder<0> const builder;

};


// FIXME: The syntax for using this isn't great.
template<Py_ssize_t LEN>
class Tuple::Builder
{
public:

  Builder(Builder<LEN - 1> last, baseref&& ref) 
    : last_(last),
      obj_(ref.release()) 
  {}

  ~Builder() { assert(obj_ == nullptr); }

  /**
   * Takes 'ref' to append the end of the tuple.
   */
  auto operator<<(baseref&& ref) const
  {
    return Builder<LEN + 1>(*this, std::move(ref));
  }

  /**
   * Builds the tuple.
   */
  operator ref<Tuple>()
  {
    auto tuple = ref<Tuple>::take(PyTuple_New(LEN));
    initialize(tuple);
    return tuple;
  }

  operator ref<Object>() { return (ref<Tuple>) *this; }

  void initialize(PyObject* tuple)
  {
    assert(obj_ != nullptr);
    last_.initialize(tuple);
    PyTuple_SET_ITEM(tuple, LEN - 1, obj_);
    obj_ = nullptr;
  }

private:

  Builder<LEN - 1> last_;
  PyObject* obj_;

};


/**
 * Base case for recursive tuple builder template.
 */
template<>
class Tuple::Builder<0>
{
public:

  Builder() {}

  auto operator<<(baseref&& ref) const 
  { 
    return Builder<1>(*this, std::move(ref)); 
  }

  operator ref<Tuple>() const 
  { 
    return ref<Tuple>::take(PyTuple_New(0)); 
  }

  void initialize(PyObject* tuple) const {}

};


//------------------------------------------------------------------------------

class List
: public Sequence
{
public:

  static bool Check(PyObject* const obj)
    { return PyList_Check(obj); }
  static ref<List> New(Py_ssize_t const len)
    { return take_not_null<List>(PyList_New(len)); }

  void initialize(Py_ssize_t const index, Object* const obj)
    { PyList_SET_ITEM(this, index, incref(obj)); }

};


//------------------------------------------------------------------------------

class Type
  : public PyTypeObject
{
public:

  static bool Check(PyObject* const obj)
    { return PyType_Check(obj); }

  Type() {}
  Type(PyTypeObject o) : PyTypeObject(o) {}

  void Ready()
    { check_zero(PyType_Ready(this)); }

};


//------------------------------------------------------------------------------

class StructSequence
  : public Object
{
public:

  Object* GetItem(Py_ssize_t index)
    { return check_not_null(PyStructSequence_GET_ITEM(this, index)); }

  void initialize(Py_ssize_t index, baseref&& ref)
    { PyStructSequence_SET_ITEM(this, index, ref.release()); }

};


//------------------------------------------------------------------------------

class StructSequenceType
  : public Type
{
public:

  static void InitType(StructSequenceType* type, PyStructSequence_Desc* desc)
    { check_zero(PyStructSequence_InitType2(type, desc)); }

#if 0
  static StructSequenceType* NewType(PyStructSequence_Desc* desc)
    { return (StructSequenceType*) check_not_null(PyStructSequence_NewType(desc)); }
#endif

  // FIXME: Doesn't work; see https://bugs.python.org/issue20066.  We can't
  // just set TPFLAGS_HEAPTYPE, as the returned type object doesn't have the
  // layout that implies.
  ref<StructSequence> New()
    { return ref<StructSequence>::take(check_not_null((PyObject*) PyStructSequence_New((PyTypeObject*) this))); }

};


//------------------------------------------------------------------------------

class Unicode
  : public Object
{
public:

  static bool Check(PyObject* obj)
    { return PyUnicode_Check(obj); }

  static auto FromString(char* utf8)
    { return ref<Unicode>::take(PyUnicode_FromString(utf8)); }
  // FIXME: Cast on const here?
  static auto FromStringAndSize(char* utf8, size_t length)
    { return ref<Unicode>::take(PyUnicode_FromStringAndSize(utf8, length)); }

  static auto from(std::string const& str)
    { return FromStringAndSize(const_cast<char*>(str.c_str()), str.length()); }

  static auto from(char character)
    { return FromStringAndSize(&character, 1); }

  char* as_utf8() { return PyUnicode_AsUTF8(this); }

  std::string as_utf8_string()
  {
    Py_ssize_t length;
    char* const utf8 = PyUnicode_AsUTF8AndSize(this, &length);
    if (utf8 == nullptr)
      throw Exception();
    else
      return std::string(utf8, length);
  }

};


template<>
inline std::ostream& operator<<(std::ostream& os, ref<Unicode>& ref)
{
  os << ref->as_utf8();
  return os;
}


inline std::string
operator+(
  std::string const& str0,
  Unicode& str1)
{
  return str0 + str1.as_utf8_string();
}


//==============================================================================

inline void baseref::clear()
{
  if (obj_ != nullptr)
    decref(obj_);
}


inline ref<Long>
Object::Long(bool const check)
{
  auto long_obj = PyNumber_Long(this);
  if (check)
    long_obj = check_not_null(long_obj);
  // FIXME: Clumsy.
  else if (long_obj == nullptr)
    Exception::Clear();
  return ref<py::Long>::take(long_obj);
}


inline long
Object::long_value()
{
  return (long) *Long();
}


inline unsigned long
Object::unsigned_long_value()
{
  return (unsigned long) *Long();
}


inline optional<long>
Object::maybe_long_value()
{
  auto obj = PyNumber_Long(this);
  if (obj == nullptr) {
    Exception::Clear();
    return {};
  }
  else {
    auto long_obj = ref<py::Long>::take(obj);
    return (long) *long_obj;
  }
}


inline ref<Float>
Object::Float()
{
  return ref<py::Float>::take(check_not_null(PyNumber_Float(this)));
}


inline double
Object::double_value()
{
  return (double) *Float();
}


inline optional<double>
Object::maybe_double_value()
{
  auto obj = PyNumber_Float(this);
  if (obj == nullptr) {
    Exception::Clear();
    return {};
  }
  else {
    auto float_obj = ref<py::Float>::take(obj);
    return (double) *float_obj;
  }
}


//==============================================================================

namespace Arg {

inline void
ParseTuple(
  Tuple* const args,
  char const* const format,
  ...)
{
  va_list vargs;
  va_start(vargs, format);
  auto result = PyArg_VaParse(args, (char*) format, vargs);
  va_end(vargs);
  check_true(result);
}


inline void ParseTupleAndKeywords(
    Tuple* args, Dict* kw_args, 
    char const* format, char const* const* keywords, ...)
{
  va_list vargs;
  va_start(vargs, keywords);
  auto result = PyArg_VaParseTupleAndKeywords(
      args, kw_args, (char*) format, (char**) keywords, vargs);
  va_end(vargs);
  check_true(result);
}


}  // namespace Arg

//==============================================================================

class ExtensionType
  : public Object
{
public:

};


//==============================================================================

// Buffer objects
// See https://docs.python.org/3/c-api/buffer.html.

/**
 * A unique reference view to a buffer object.
 *
 * Supports move semantics only; no copy.
 */
class BufferRef
{
public:

  /**
   * Creates a buffer view of an object.  The ref holds a reference to the
   * object.
   */
  BufferRef(PyObject* obj, int flags)
  {
    if (PyObject_GetBuffer(obj, &buffer_, flags) != 0)
      throw Exception();
    assert(buffer_.obj != nullptr);
  }

  BufferRef(Py_buffer&& buffer)
    : buffer_(buffer)
  {}

  BufferRef(BufferRef&& ref)
    : buffer_(ref.buffer_)
  {
    ref.buffer_.obj = nullptr;
  }

  ~BufferRef()
  {
    // Only releases if buffer_.obj is not null.
    PyBuffer_Release(&buffer_);
    assert(buffer_.obj == nullptr);
  }

  BufferRef(BufferRef const&) = delete;
  void operator=(BufferRef const&) = delete;

  Py_buffer* operator->() { return &buffer_; }

private:

  Py_buffer buffer_;

};


//==============================================================================
// Inline methods

inline ref<Object> 
Object::CallObject(Tuple* args)
{ 
  return ref<Object>::take(check_not_null(PyObject_CallObject(this, args))); 
}


//------------------------------------------------------------------------------
// Exception translation
//------------------------------------------------------------------------------
// Inspired by exception translation in boost::python.

/*
 * Common base class for <TranslateException>.  Don't use directly.
 */
class ExceptionTranslator
{
public:

  static void 
  translate()
  {
    if (head_ == nullptr)
      // No translations registerd.
      throw;
    else
      // Start at the head of the list.
      head_->translate1();
  }

private:

  ExceptionTranslator() = default;

  // Head of the list of registered translations.
  static ExceptionTranslator* head_;

  // Next registered translation, i.e. link field in the list of translations.
  ExceptionTranslator* next_ = nullptr;

  virtual void translate1() const = 0;

  template<class EXCEPTION>
  friend class TranslateException;

};


template<class EXCEPTION>
class TranslateException
: ExceptionTranslator
{
public:

  /*
   * Registers an exception translation.
   *
   * Registers translation from C++ exception class `EXCEPTION` to Python 
   * exception class `exception`.
   *
   * The `EXCEPTION` class must have a `what()` method that returns the
   * exception message.
   */
  static void
  to(
    PyObject* const exception)
  {
    // Register an instance of ourselves at the front of the list.
    auto const translator = new TranslateException(exception);
    translator->next_ = head_;
    head_ = translator;
  }

private:

  /*
   * Translates the current C++ exception.
   * 
   * - If the exception matches `EXCEPTION`, throw <Exception> to raise the
   *   registered Python exception class.
   * - Otherwise, call <translate1()> on the next exception translator in the
   *   list, with the previous C++ exception still in flight.
   * - If this is the last exception translator on the list, translate to
   *   a Python `RuntimeError`.
   */
  virtual void 
  translate1()
    const
  {
    try {
      throw;
    }
    catch (EXCEPTION exc) {
      throw Exception(exception_, exc.what());
    }
    catch (...) {
      if (next_ == nullptr)
        throw Exception(PyExc_RuntimeError, "untranslated C++ exception");
      else
        next_->translate1();
    }
  }

  TranslateException(
    PyObject* exception)
  : exception_(exception)
  {
  }

  // The Python exception type into which to translate `EXCEPTION` instances.
  PyObject* const exception_;

};


//==============================================================================

template<class CLASS>
using
BinaryfuncPtr
  = ref<Object> (*)(CLASS* self, Object* other, bool right);

template<class CLASS>
using
DestructorPtr
  = void (*)(CLASS* self);

template<class CLASS>
using 
InitprocPtr
  = void (*)(CLASS* self, Tuple* args, Dict* kw_args);

template<class CLASS>
using
ReprfuncPtr
  = ref<Unicode> (*)(CLASS* self);

template<class CLASS>
using
RichcmpfuncPtr
  = ref<Object> (*)(CLASS* self, Object* other, int comparison);

template<class CLASS>
using
HashfuncPtr
  = Py_hash_t (*)(CLASS* self);

template<class CLASS>
using MethodPtr = ref<Object> (*)(CLASS* self, Tuple* args, Dict* kw_args);

using StaticMethodPtr = ref<Object> (*)(Tuple* args, Dict* kw_args);

using ClassMethodPtr = ref<Object> (*)(PyTypeObject* class_, Tuple* args, Dict* kw_args);


/**
 * Wraps a binaryfunc.
 */
template<class CLASS, BinaryfuncPtr<CLASS> FUNCTION>
PyObject*
wrap(
  PyObject* lhs,
  PyObject* rhs)
{
  ref<Object> result;
  try {
    try {
      if (CLASS::Check(lhs)) 
        result = FUNCTION(
          static_cast<CLASS*>(lhs), static_cast<Object*>(rhs), false);
      else if (CLASS::Check(rhs))
        result = FUNCTION(
          static_cast<CLASS*>(rhs), static_cast<Object*>(lhs), true);
      else
        result = not_implemented_ref();
    }
    catch (Exception) {
      return nullptr;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception) {
    return nullptr;
  }
  assert(result != nullptr);
  return result.release();
}


/**
 * Wraps a destructor.
 */
template<class CLASS, DestructorPtr<CLASS> FUNCTION>
void
wrap(
  PyObject* self)
{
  // tp_dealloc should preserve exception state; maybe wrap with PyErr_Fetch()
  // and PyErr_restore()?
  try {
    try {
      FUNCTION(static_cast<CLASS*>(self));
    }
    catch (Exception) {
      return;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception exc) {
    // Eat it.
  }
}


/**
 * Wraps an initproc.
 */
template<class CLASS, InitprocPtr<CLASS> FUNCTION>
int
wrap(
  PyObject* self,
  PyObject* args,
  PyObject* kw_args)
{
  try {
    try {
      FUNCTION(
        static_cast<CLASS*>(self),
        static_cast<Tuple*>(args),
        static_cast<Dict*>(kw_args));
    }
    catch (Exception) {
      return -1;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception) {
    return -1;
  }
  return 0;
}


/**
 * Wraps a reprfunc.
 */
template<class CLASS, ReprfuncPtr<CLASS> FUNCTION>
PyObject*
wrap(
  PyObject* self)
{
  ref<Unicode> result;
  try {
    try {
      result = FUNCTION(static_cast<CLASS*>(self));
    }
    catch (Exception) {
      return nullptr;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception) {
    return nullptr;
  }
  assert(result != nullptr);
  return result.release();
}


/**
 * Wraps a richcmpfunc.
 */
template<class CLASS, RichcmpfuncPtr<CLASS> FUNCTION>
PyObject*
wrap(
  PyObject* const self,
  PyObject* const other,
  int const comparison)
{
  ref<Object> result;
  try {
    try {
      result = FUNCTION(
        static_cast<CLASS*>(self), 
        static_cast<Object*>(other), 
        comparison);
    }
    catch (Exception) {
      return nullptr;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception) {
    return nullptr;
  }
  assert(result != nullptr);
  return result.release();
}


/*
 * Wraps a hashfunc.
 */
template<class CLASS, HashfuncPtr<CLASS> HASHFUNC>
Py_hash_t
wrap(
  PyObject* self)
{
  Py_hash_t result = -1;
  try {
    try {
      result = HASHFUNC(static_cast<CLASS*>(self));
    }
    catch (Exception) {
      return -1;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception) {
    return -1;
  }
  assert(result != -1);
  return result;
}


/**
 * Wraps a method that takes args and kw_args and returns an object.
 */
template<class CLASS, MethodPtr<CLASS> METHOD>
PyObject* wrap(PyObject* self, PyObject* args, PyObject* kw_args)
{
  ref<Object> result;
  try {
    try {
      result = METHOD(
        reinterpret_cast<CLASS*>(self),
        reinterpret_cast<Tuple*>(args),
        reinterpret_cast<Dict*>(kw_args));
    }
    catch (Exception) {
      return nullptr;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception) {
    return nullptr;
  }
  assert(result != nullptr);
  return result.release();
}


template<StaticMethodPtr METHOD>
PyObject* 
wrap_static_method(
  PyObject* /* unused */, 
  PyObject* args, 
  PyObject* kw_args)
{
  ref<Object> result;
  try {
    try {
      result = METHOD(
        reinterpret_cast<Tuple*>(args),
        reinterpret_cast<Dict*>(kw_args));
    }
    catch (Exception) {
      return nullptr;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception) {
    return nullptr;
  }
  assert(result != nullptr);
  return result.release();
}


template<ClassMethodPtr METHOD>
PyObject* 
wrap_class_method(
  PyObject* class_,
  PyObject* args, 
  PyObject* kw_args)
{
  ref<Object> result;
  try {
    try {
      result = METHOD(
        reinterpret_cast<PyTypeObject*>(class_),
        reinterpret_cast<Tuple*>(args),
        reinterpret_cast<Dict*>(kw_args));
    }
    catch (Exception) {
      return nullptr;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception) {
    return nullptr;
  }
  assert(result != nullptr);
  return result.release();
}


//------------------------------------------------------------------------------

template<class CLASS>
class Methods
{
public:

  Methods() : done_(false) {}

  template<MethodPtr<CLASS> METHOD>
  Methods& add(char const* name, char const* doc=nullptr)
  {
    assert(name != nullptr);
    assert(!done_);
    methods_.push_back({
      name,
      (PyCFunction) wrap<CLASS, METHOD>,
      METH_VARARGS | METH_KEYWORDS,
      doc
    });
    return *this;
  }

  template<StaticMethodPtr METHOD>
  Methods& add_static(char const* const name, char const* const doc=nullptr)
  {
    assert(name != nullptr);
    assert(!done_);
    methods_.push_back({
      name,
      (PyCFunction) wrap_static_method<METHOD>,
      METH_VARARGS | METH_KEYWORDS | METH_STATIC,
      doc
    });
    return *this;
  }

  template<ClassMethodPtr METHOD>
  Methods& add_class(char const* const name, char const* const doc=nullptr)
  {
    assert(name != nullptr);
    assert(!done_);
    methods_.push_back({
      name,
      (PyCFunction) wrap_class_method<METHOD>,
      METH_VARARGS | METH_KEYWORDS | METH_CLASS,
      doc
    });
    return *this;
  }

  operator PyMethodDef*()
  {
    if (!done_) {
      // Add the sentry.
      methods_.push_back({nullptr, nullptr, 0, nullptr});
      done_ = true;
    }
    return &methods_[0];
  }

private:

  bool done_;
  std::vector<PyMethodDef> methods_;

};


//------------------------------------------------------------------------------

template<class CLASS>
using GetPtr = ref<Object> (*)(CLASS* self, void* closure);


template<class CLASS, GetPtr<CLASS> METHOD>
PyObject* wrap_get(PyObject* self, void* closure)
{
  ref<Object> result;
  try {
    try {
      result = METHOD(reinterpret_cast<CLASS*>(self), closure); 
    }
    catch (Exception) {
      return nullptr;
    }
    catch (...) {
      ExceptionTranslator::translate();
    }
  }
  catch (Exception) {
    return nullptr;
  }
  assert(result != nullptr);
  return result.release();
}


template<class CLASS>
class GetSets
{
public:

  GetSets() : done_(false) {}

  template<GetPtr<CLASS> METHOD>
  GetSets& add_get(char const* name, char const* doc=nullptr, 
                   void* closure=nullptr)
  {
    assert(name != nullptr);
    assert(!done_);
    getsets_.push_back({
      (char*)   name,
      (getter)  wrap_get<CLASS, METHOD>,
      (setter)  nullptr,
      (char*)   doc,
      (void*)   closure,
    });
    return *this;
  }

  operator PyGetSetDef*()
  {
    if (!done_) {
      // Add the sentry.
      getsets_.push_back({nullptr, nullptr, nullptr, nullptr, nullptr});
      done_ = true;
    }
    return &getsets_[0];
  }

private:

  bool done_;
  std::vector<PyGetSetDef> getsets_;

};


//==============================================================================

inline ref<Object>
import(const char* module_name, const char* name)
{
  return Module::ImportModule(module_name)->GetAttrString(name);
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

