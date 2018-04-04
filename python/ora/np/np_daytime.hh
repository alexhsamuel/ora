#include <Python.h>

#include "ora/lib/mem.hh"
#include "ora.hh"
#include "py.hh"
#include "np_types.hh"
#include "numpy.hh"
#include "PyDaytime.hh"

namespace ora {
namespace py {

using namespace py;
using namespace py::np;

//------------------------------------------------------------------------------

class DaytimeAPI
{
private:

  static uint64_t constexpr MAGIC = 0x737865c3443a5a50;
  uint64_t const magic_ = MAGIC;

  static DaytimeAPI*
  get(
    PyArray_Descr* const dtype)
  {
    // Make an attempt to confirm that this is one of our dtypes.
    if (dtype->kind == 'V' && dtype->type == 'j') {
      auto const api = reinterpret_cast<DaytimeAPI*>(dtype->c_metadata);
      if (api != nullptr && api->magic_ == MAGIC)
        return api;
    }
    return nullptr;
  }

public:

  virtual ~DaytimeAPI() {}

  /*
   * Converts a daytick to a daytime, and stores it at an address. 
   */
  virtual void          from_daytick(ora::Daytick, void*) const = 0;

  /*
   * Returns the daytick for a daytime at an address.
   */
  virtual Daytick       get_daytick(void*) const = 0;

  static bool check(PyArray_Descr* const descr)
    { return get(descr) != nullptr; }

  static DaytimeAPI*
  from(
    PyArray_Descr* const dtype)
  {
    auto const api = get(dtype);
    if (api == nullptr)
      throw TypeError("not an ora daytime dtype");
    else
      return api;
  }

};


template<class PYDAYTIME>
class DaytimeDtype
{
public:

  using Daytime = typename PYDAYTIME::Daytime;

  /*
   * Returns the singletone descriptor / dtype object.
   */
  static PyArray_Descr* get();

  /*
   * Adds the dtype object.
   */
  static void           add(Module*);

private:

  // FIXME: Wrap these.
  static void           copyswap(Daytime*, Daytime const*, int, PyArrayObject*);
  static void           copyswapn(Daytime*, npy_intp, Daytime const*, npy_intp, npy_intp, int, PyArrayObject*);
  static Object*        getitem(Daytime const*, PyArrayObject*);
  static int            setitem(Object*, Daytime*, PyArrayObject*);
  static int            compare(Daytime const*, Daytime const*, PyArrayObject*);

  // Wrap days_after and days_before to accept int64 args.
  static Daytime add(Daytime const daytime, double const seconds)
    { return ora::daytime::nex::seconds_after(daytime, seconds); }
  static Daytime subtract_before(Daytime const daytime, double const seconds)
    { return ora::daytime::nex::seconds_before(daytime, seconds); }
  static double subtract_between(Daytime const daytime1, Daytime const daytime0) 
    { return ora::daytime::nex::seconds_between(daytime0, daytime1); }

  class API
  : public DaytimeAPI
  {
  public:

    virtual ~API() {}

    virtual void 
    from_daytick(
      ora::Daytick daytick, 
      void* daytime_ptr) 
      const override
    { 
      *reinterpret_cast<Daytime*>(daytime_ptr) 
        = ora::daytime::nex::from_daytick<Daytime>(daytick); 
    }

    virtual Daytick get_daytick(void* const daytime_ptr) const override
      { return ora::daytime::nex::get_daytick(*reinterpret_cast<Daytime*>(daytime_ptr)); }

  };

  static PyArray_Descr* descr_;

};


template<class PYDAYTIME>
PyArray_Descr*
DaytimeDtype<PYDAYTIME>::get()
{
  if (descr_ == nullptr) {
    auto const arr_funcs = new PyArray_ArrFuncs;
    PyArray_InitArrFuncs(arr_funcs);
    arr_funcs->copyswap         = (PyArray_CopySwapFunc*) copyswap;
    arr_funcs->copyswapn        = (PyArray_CopySwapNFunc*) copyswapn;
    arr_funcs->getitem          = (PyArray_GetItemFunc*) getitem;
    arr_funcs->setitem          = (PyArray_SetItemFunc*) setitem;
    arr_funcs->compare          = (PyArray_CompareFunc*) compare;

    descr_ = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    descr_->typeobj         = incref(&PYDAYTIME::type_);
    descr_->kind            = 'V';
    descr_->type            = 'j';  // FIXME
    descr_->byteorder       = '=';
    descr_->flags           = 0;
    descr_->type_num        = 0;
    descr_->elsize          = sizeof(Daytime);
    descr_->alignment       = alignof(Daytime);
    descr_->subarray        = nullptr;
    descr_->fields          = nullptr;
    descr_->names           = nullptr;
    descr_->f               = arr_funcs;
    descr_->metadata        = nullptr;
    descr_->c_metadata      = (NpyAuxData*) new API();
    descr_->hash            = -1;

    if (PyArray_RegisterDataType(descr_) < 0)
      throw py::Exception();
  }

  return descr_;
}


template<class PYDAYTIME>
void
DaytimeDtype<PYDAYTIME>::add(
  Module* const module)
{
  auto const np_module = Module::ImportModule("numpy");

  // Build or get the dtype.
  auto const dtype = DaytimeDtype<PYDAYTIME>::get();

  // Add the dtype as a class attribute.
  auto const dict = (Dict*) dtype->typeobj->tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("dtype", (Object*) dtype);

  // Add ufuncs.
  // FIXME

  Comparisons<Daytime, ora::daytime::nex::equal, ora::daytime::nex::before>
    ::register_loops(dtype->type_num);

  // Add ufunc loops.
  create_or_get_ufunc(np_module, "add", 2, 1)->add_loop_2(
    dtype->type_num, NPY_FLOAT64, dtype->type_num,
    ufunc_loop_2<Daytime, double, Daytime, add>);
  create_or_get_ufunc(np_module, "subtract", 2, 1)->add_loop_2(
    dtype->type_num, NPY_FLOAT64, dtype->type_num,
    ufunc_loop_2<Daytime, double, Daytime, subtract_before>);
  create_or_get_ufunc(np_module, "subtract", 2, 1)->add_loop_2(
    dtype->type_num, dtype->type_num, NPY_FLOAT64,
    ufunc_loop_2<Daytime, Daytime, double, subtract_between>);
}


//------------------------------------------------------------------------------
// numpy array functions

template<class PYDAYTIME>
void
DaytimeDtype<PYDAYTIME>::copyswap(
  Daytime* const dst,
  Daytime const* const src,
  int const swap,
  PyArrayObject* const arr)
{
  if (swap)
    copy_swapped<sizeof(Daytime)>(src, dst);
  else
    copy<sizeof(Daytime)>(src, dst);
}


template<class PYDAYTIME>
void 
DaytimeDtype<PYDAYTIME>::copyswapn(
  Daytime* const dst, 
  npy_intp const dst_stride, 
  Daytime const* const src, 
  npy_intp const src_stride, 
  npy_intp const n, 
  int const swap, 
  PyArrayObject* const arr)
{
  // FIXME: Abstract this out.
  if (src_stride == 0) {
    // Swapper or unswapped fill.  Optimize this special case.
    Daytime daytime;
    if (swap) 
      copy_swapped<sizeof(Daytime)>(src, &daytime);
    else
      daytime = *src;
    Daytime* d = dst;
    for (npy_intp i = 0; i < n; ++i) {
      *d = daytime;
      d = (Daytime*) (((char*) d) + dst_stride);
    }
  }
  else {
    char const* s = (char const*) src;
    char* d = (char*) dst;
    if (swap) 
      for (npy_intp i = 0; i < n; ++i) {
        copy_swapped<sizeof(Daytime)>(s, d);
        s += src_stride;
        d += dst_stride;
      }
    else 
      for (npy_intp i = 0; i < n; ++i) {
        copy<sizeof(Daytime)>(s, d);
        s += src_stride;
        d += dst_stride;
      }
  }
}


template<class PYDAYTIME>
Object*
DaytimeDtype<PYDAYTIME>::getitem(
  Daytime const* const data,
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "getitem\n";
  return PYDAYTIME::create(*data).release();
}


template<class PYDAYTIME>
int
DaytimeDtype<PYDAYTIME>::setitem(
  Object* const item,
  Daytime* const data,
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "setitem\n";
  try {
    *data = convert_to_daytime<Daytime>(item);
  }
  catch (Exception) {
    return -1;
  }
  return 0;
}


template<class PYDAYTIME>
int 
DaytimeDtype<PYDAYTIME>::compare(
  Daytime const* const d0, 
  Daytime const* const d1, 
  PyArrayObject* const /* arr */)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "compare\n";
  // Invalid compares least, then missing, then other daytimes.
  return 
      d0->is_invalid() ? -1
    : d1->is_invalid() ?  1
    : d0->is_missing() ? -1
    : d1->is_missing() ?  1
    : *d0 < *d1        ? -1 
    : *d0 > *d1        ?  1 
    : 0;
}


//------------------------------------------------------------------------------

template<class PYDAYTIME>
PyArray_Descr*
DaytimeDtype<PYDAYTIME>::descr_
  = nullptr;

//------------------------------------------------------------------------------
// Accessories

/*
 * Attempts to convert `arg` to a aytime array.
 *
 * If it isn't one already, attempts to convert it using the default daytime
 * dtype.
 */
inline ref<Array>
to_daytime_array(
  Object* const arg)
{
  if (Array::Check(arg)) {
    // It's an array.  Check its dtype.
    Array* const arr = reinterpret_cast<Array*>(arg);
    if (DaytimeAPI::check(arr->descr()))
      return ref<Array>::of(arr);
  }

  // Convert to an array of the default time dtype.
  auto const def = DaytimeDtype<PyDaytimeDefault>::get();
  return Array::FromAny(arg, def, 0, 0, NPY_ARRAY_BEHAVED);
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

