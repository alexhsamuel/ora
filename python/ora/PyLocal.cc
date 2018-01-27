#include "py.hh"
#include "PyLocal.hh"

using std::string;
using namespace std::string_literals;

namespace ora {
namespace py {

//------------------------------------------------------------------------------

void
PyLocal::add_to(
  Module& module,
  string const& name)
{
  // Construct the type struct.
  type_ = build_type(string{module.GetName()} + "." + name);
  // Hand it to Python.
  type_.Ready();

  // Add the type to the module.
  module.add(&type_);
}


//------------------------------------------------------------------------------
// Data members
//------------------------------------------------------------------------------

Type
PyLocal::type_;

//------------------------------------------------------------------------------
// Standard type methods
//------------------------------------------------------------------------------

void
PyLocal::tp_dealloc(
  PyLocal* const self)
{
  self->ob_type->tp_free(self);
}


ref<Unicode>
PyLocal::tp_repr(
  PyLocal* const self)
{
  return Unicode::from(
    "LocalTime("s
    + self->date_->Repr()->as_utf8_string() 
    + ", " 
    + self->daytime_->Repr()->as_utf8_string() 
    + ")");
}


ref<Unicode>
PyLocal::tp_str(
  PyLocal* const self)
{
  return Unicode::from(
    self->date_->Str()->as_utf8_string() 
    + " " 
    + self->daytime_->Str()->as_utf8_string());
}


ref<Object>
PyLocal::tp_richcompare(
  PyLocal* const self,
  Object* const other,
  int const comparison)
{
  if (Sequence::Check(other)) {
    auto const seq = cast<Sequence>(other);
    if (seq->Length() == 2) {
      auto& d0 = self->date_;
      auto& y0 = self->daytime_;
      auto const d1 = seq->GetItem(0);
      auto const y1 = seq->GetItem(1);

      bool result;
      switch (comparison) {
      case Py_EQ: result = d0->eq(d1) && y0->eq(y1); break;
      case Py_NE: result = d0->ne(d1) || y0->ne(y1); break;
      case Py_LT: result = d0->lt(d1) || (d0->eq(d1) && y0->lt(y1)); break;
      case Py_LE: result = d0->lt(d1) || (d0->eq(d1) && y0->le(y1)); break;
      case Py_GT: result = d0->gt(d1) || (d0->eq(d1) && y0->gt(y1)); break;
      case Py_GE: result = d0->gt(d1) || (d0->eq(d1) && y0->ge(y1)); break;
      }
      return Bool::from(result);
    }
  }

  return not_implemented_ref();
}


void
PyLocal::tp_init(
  PyLocal* const self,
  Tuple* const args,
  Dict* const kw_args)
{
  // FIXME: Accept other things too.
  Object* date;
  Object* daytime;
  Arg::ParseTuple(args, "OO", &date, &daytime);

  new(self) PyLocal(date, daytime);
}


//------------------------------------------------------------------------------
// Sequence
//------------------------------------------------------------------------------

Py_ssize_t
PyLocal::sq_length(
  PyLocal* const self)
{
  return 2;
}


ref<Object>
PyLocal::sq_item(
  PyLocal* const self,
  Py_ssize_t const index)
{
  if (index == 0)
    return self->date_.inc();
  else if (index == 1)
    return self->daytime_.inc();
  else
    throw IndexError("index out of range");
}


PySequenceMethods const 
PyLocal::tp_as_sequence = {
  (lenfunc)         sq_length,                          // sq_length
  (binaryfunc)      nullptr,                            // sq_concat
  (ssizeargfunc)    nullptr,                            // sq_repeat
  (ssizeargfunc)    wrap<PyLocal, sq_item>,             // sq_item
  (void*)           nullptr,                            // was_sq_slice
  (ssizeobjargproc) nullptr,                            // sq_ass_item
  (void*)           nullptr,                            // was_sq_ass_slice
  (objobjproc)      nullptr,                            // sq_contains
  (binaryfunc)      nullptr,                            // sq_inplace_concat
  (ssizeargfunc)    nullptr,                            // sq_inplace_repeat
};


//------------------------------------------------------------------------------
// Getsets
//------------------------------------------------------------------------------

ref<Object>
PyLocal::get_date(
  PyLocal* const self,
  void* /* closure */)
{
  return self->date_.inc();
}


ref<Object>
PyLocal::get_daytime(
  PyLocal* const self,
  void* /* closure */)
{
  return self->daytime_.inc();
}


GetSets<PyLocal>
PyLocal::tp_getsets_
  = GetSets<PyLocal>()
    .template add_get<get_date>             ("date")
    .template add_get<get_daytime>          ("daytime")
  ;


//------------------------------------------------------------------------------
// Type object
//------------------------------------------------------------------------------

Type
PyLocal::build_type(
  string const& type_name)
{
  return PyTypeObject{
    PyVarObject_HEAD_INIT(nullptr, 0)
    (char const*)         strdup(type_name.c_str()),      // tp_name
    (Py_ssize_t)          sizeof(PyLocal),                // tp_basicsize
    (Py_ssize_t)          0,                              // tp_itemsize
    (destructor)          wrap<PyLocal, tp_dealloc>,      // tp_dealloc
    (printfunc)           nullptr,                        // tp_print
    (getattrfunc)         nullptr,                        // tp_getattr
    (setattrfunc)         nullptr,                        // tp_setattr
                          nullptr,                        // tp_reserved
    (reprfunc)            wrap<PyLocal, tp_repr>,         // tp_repr
    (PyNumberMethods*)    nullptr,                        // tp_as_number
    (PySequenceMethods*)  &tp_as_sequence,                // tp_as_sequence
    (PyMappingMethods*)   nullptr,                        // tp_as_mapping
    (hashfunc)            nullptr,                        // tp_hash
    (ternaryfunc)         nullptr,                        // tp_call
    (reprfunc)            wrap<PyLocal, tp_str>,          // tp_str
    (getattrofunc)        nullptr,                        // tp_getattro
    (setattrofunc)        nullptr,                        // tp_setattro
    (PyBufferProcs*)      nullptr,                        // tp_as_buffer
    (unsigned long)       Py_TPFLAGS_DEFAULT
                          | Py_TPFLAGS_BASETYPE,          // tp_flags
    (char const*)         nullptr,                        // tp_doc
    (traverseproc)        nullptr,                        // tp_traverse
    (inquiry)             nullptr,                        // tp_clear
    (richcmpfunc)         wrap<PyLocal, tp_richcompare>,  // tp_richcompare
    (Py_ssize_t)          0,                              // tp_weaklistoffset
    (getiterfunc)         nullptr,                        // tp_iter
    (iternextfunc)        nullptr,                        // tp_iternext
    (PyMethodDef*)        nullptr,                        // tp_methods
    (PyMemberDef*)        nullptr,                        // tp_members
    (PyGetSetDef*)        tp_getsets_,                    // tp_getset
    (_typeobject*)        nullptr,                        // tp_base
    (PyObject*)           nullptr,                        // tp_dict
    (descrgetfunc)        nullptr,                        // tp_descr_get
    (descrsetfunc)        nullptr,                        // tp_descr_set
    (Py_ssize_t)          0,                              // tp_dictoffset
    (initproc)            wrap<PyLocal, tp_init>,         // tp_init
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

}  // namespace py
}  // namespace ora

