#pragma once

#include <cassert>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include <Python.h>

//------------------------------------------------------------------------------

namespace py {

class Long;
class Object;
class Tuple;
class Unicode;

// FIXME: Remove this.
constexpr PyGetSetDef GETSETDEF_END
    {nullptr, nullptr, nullptr, nullptr, nullptr};

//------------------------------------------------------------------------------

class Exception
{
public:

  Exception() {}
  
  Exception(PyObject* exception, char const* message)
  {
    PyErr_SetString(exception, message);
  }

  template<typename A>
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

  template<typename A>
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


/**
 * Raises 'Exception' if value is not true.
 */
inline void check_true(int value)
{
  if (value == 0)
    throw Exception();
}


//------------------------------------------------------------------------------

template<typename T>
inline T* cast(PyObject* obj)
{
  assert(T::Check(obj));  // FIXME: TypeError?
  return static_cast<T*>(obj);
}


//------------------------------------------------------------------------------

inline PyObject* incref(PyObject* obj)
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


template<typename T>
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
  template<typename U>
  ref(ref<U>&& ref)
    : baseref(ref.release()) {}

  void operator=(ref<T>&& ref)
    { clear(); obj_ = ref.release(); }

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


//==============================================================================

class Object
  : public PyObject
{
public:

  static bool Check(PyObject* obj)
    { return true; }

  auto Length()
    { return PyObject_Length(this); }
  auto Repr()
    { return ref<Unicode>::take(PyObject_Repr(this)); }
  auto Str()
    { return ref<Unicode>::take(PyObject_Str(this)); }

  ref<py::Long> Long();

  long long_value();

};


template<typename T>
inline std::ostream& operator<<(std::ostream& os, ref<T>& ref)
{
  os << ref->Str()->as_utf8();
  return os;
}


//------------------------------------------------------------------------------

class Dict
  : public Object
{
public:

  static bool Check(PyObject* obj)
    { return PyDict_Check(obj); }

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

class Long
  : public Object
{
public:

  static bool Check(PyObject* obj)
    { return PyLong_Check(obj); }
  static auto FromLong(long val)
    { return ref<Long>::take(PyLong_FromLong(val)); }

  operator long()
    { return PyLong_AsLong(this); }

};


//------------------------------------------------------------------------------

class Float
  : public Object
{
public:

  static bool Check(PyObject* obj)
    { return PyFloat_Check(obj); }
  static auto FromDouble(double val)
    { return ref<Float>::take(PyFloat_FromDouble(val)); }

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

class Tuple
  : public Object
{
public:

  static bool Check(PyObject* obj)
    { return PyTuple_Check(obj); }

  static auto New(Py_ssize_t len)
    { return ref<Tuple>::take(PyTuple_New(len)); }

  void initialize(Py_ssize_t index, baseref&& ref)
  {
    PyTuple_SET_ITEM(this, index, ref.release());
  }

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

class Type
  : public PyTypeObject
{
public:

  Type() {}
  Type(PyTypeObject o) : PyTypeObject(o) {}

  void Ready()
    { check_zero(PyType_Ready(this)); }

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


//==============================================================================

inline void baseref::clear()
{
  if (obj_ != nullptr)
    decref(obj_);
}


inline ref<Long>
Object::Long()
{
  // FIXME: Check errors.
  return ref<py::Long>::take(PyNumber_Long(this));
}


inline long
Object::long_value()
{
  return (long) *Long();
}


//==============================================================================

namespace Arg {

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

  PyObject_HEAD

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

template<typename CLASS>
using MethodPtr = ref<Object> (*)(CLASS* self, Tuple* args, Dict* kw_args);


/**
 * Wraps a method that takes args and kw_args and returns an object.
 */
template<typename CLASS, MethodPtr<CLASS> METHOD>
PyObject* wrap(PyObject* self, PyObject* args, PyObject* kw_args)
{
  ref<Object> result;
  try {
    result = METHOD(
      reinterpret_cast<CLASS*>(self),
      reinterpret_cast<Tuple*>(args),
      reinterpret_cast<Dict*>(kw_args));
  }
  catch (Exception) {
    return nullptr;
  }
  assert(result != nullptr);
  return result.release();
}


template<typename CLASS>
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

template<typename CLASS>
using GetPtr = ref<Object> (*)(CLASS* self, void* closure);


template<typename CLASS, GetPtr<CLASS> METHOD>
PyObject* wrap_get(PyObject* self, void* closure)
{
  ref<Object> result;
  try {
    result = METHOD(reinterpret_cast<CLASS*>(self), closure); 
  }
  catch (Exception) {
    return nullptr;
  }
  assert(result != nullptr);
  return result.release();
}


template<typename CLASS>
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


//------------------------------------------------------------------------------

}  // namespace py

