#include <algorithm>
#include <Python.h>

#include "ora/lib/mem.hh"
#include "ora.hh"
#include "py.hh"
#include "np_types.hh"
#include "numpy.hh"
#include "PyDate.hh"

// FIXME: Check GIL flags.

namespace ora {
namespace py {

using namespace np;

//------------------------------------------------------------------------------

// FIXME: For debugging; remove this, eventually.
bool constexpr
PRINT_ARR_FUNCS
  = false;


class DateAPI
{
private:

  static uint64_t constexpr MAGIC = 0x231841de2fe33131;
  uint64_t const magic_ = MAGIC;

  static DateAPI*
  get(
    PyArray_Descr* const dtype)
  {
    // Make an attempt to confirm that this is one of our dtypes.
    if (dtype->kind == 'V' && dtype->type == 'j') {
      auto const api = reinterpret_cast<DateAPI*>(dtype->c_metadata);
      if (api != nullptr && api->magic_ == MAGIC)
        return api;
    }
    return nullptr;
  }


public:

  virtual ~DateAPI() {}

  // FIXME: Add date_from_iso_date().

  /*
   * Converts a datenum to a date, and stores it at an address.  Returns true
   * if the date is valid.
   */
  virtual bool        from_datenum(ora::Datenum, void*) const = 0;

  /*
   * Returns the datenum for a date at an address.
   */
  virtual Datenum     get_datenum(void*) const = 0;

  virtual ref<Object> function_date_from_ordinal_date(Array*, Array*) = 0;
  virtual ref<Object> function_date_from_week_date(Array*, Array*, Array*) = 0;
  virtual ref<Object> function_date_from_ymd(Array*, Array*, Array*) = 0;
  virtual ref<Object> function_date_from_ymdi(Array*) = 0;

  static bool check(PyArray_Descr* const descr)
    { return get(descr) != nullptr; }

  static DateAPI*
  from(
    PyArray_Descr* const dtype)
  {
    auto const api = get(dtype);
    if (api == nullptr)
      throw TypeError("not an ora date dtype");
    else
      return api;
  }

};


template<class PYDATE>
class DateDtype
{
public:

  using Date = typename PYDATE::Date;

  /*
   * Returns the singleton descriptor / dtype object.
   */
  static PyArray_Descr* get();

  /*
   * Adds the dtype object to the Python type object as the `dtype` attribute.
   */
  static void           add(Module*);

private:

  // FIXME: Wrap these.
  static void           copyswap(Date*, Date const*, int, PyArrayObject*);
  static void           copyswapn(Date*, npy_intp, Date const*, npy_intp, npy_intp, int, PyArrayObject*);
  static Object*        getitem(Date const*, PyArrayObject*);
  static int            setitem(Object*, Date*, PyArrayObject*);
  static int            compare(Date const*, Date const*, PyArrayObject*);

  static void           cast_from_object(Object* const*, Date*, npy_intp, void*, void*);

  static npy_bool is_valid(Date const date)
    { return date.is_valid() ? NPY_TRUE : NPY_FALSE; }

  // Wrap days_after and days_before to accept int64 args.
  static Date add(Date const date, int64_t const days)
    { return ora::date::nex::days_after(date, (int32_t) days); }
  static Date subtract_before(Date const date, int64_t const days)
    { return ora::date::nex::days_before(date, (int32_t) days); }
  static int32_t subtract_between(Date const date1, Date const date0) 
    { return ora::date::nex::days_between(date0, date1); }

  class API
  : public DateAPI
  {
  public:

    virtual ~API() {}

    virtual bool 
    from_datenum(
      ora::Datenum const datenum, 
      void* const date_ptr) 
      const override
    { 
      auto const date = ora::date::nex::from_datenum<Date>(datenum);
      *reinterpret_cast<Date*>(date_ptr) = date;
      return date.is_valid();
    }

    virtual Datenum get_datenum(void* const date_ptr) const override
      { return ora::date::nex::get_datenum(*reinterpret_cast<Date*>(date_ptr)); }

    virtual ref<Object> function_date_from_ordinal_date(Array*, Array*) override;
    virtual ref<Object> function_date_from_week_date(Array*, Array*, Array*) override;
    virtual ref<Object> function_date_from_ymd(Array*, Array*, Array*) override;
    virtual ref<Object> function_date_from_ymdi(Array*) override;

  };

  static PyArray_Descr* descr_;

};


template<class PYDATE>
PyArray_Descr*
DateDtype<PYDATE>::get()
{
  if (descr_ == nullptr) {
    // Deliberately 'leak' this instance, as it has process lifetime.
    auto const arr_funcs = new PyArray_ArrFuncs;
    PyArray_InitArrFuncs(arr_funcs);
    arr_funcs->copyswap     = (PyArray_CopySwapFunc*) copyswap;
    arr_funcs->copyswapn    = (PyArray_CopySwapNFunc*) copyswapn;
    arr_funcs->getitem      = (PyArray_GetItemFunc*) getitem;
    arr_funcs->setitem      = (PyArray_SetItemFunc*) setitem;
    arr_funcs->compare      = (PyArray_CompareFunc*) compare;
    // FIMXE: Additional methods.

    descr_ = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    descr_->typeobj         = incref(&PYDATE::type_);
    descr_->kind            = 'V';
    descr_->type            = 'j';  // FIXME?
    descr_->byteorder       = '=';
    descr_->flags           = 0;
    descr_->type_num        = 0;
    descr_->elsize          = sizeof(Date);
    descr_->alignment       = alignof(Date);
    descr_->subarray        = nullptr;
    descr_->fields          = nullptr;
    descr_->names           = nullptr;
    descr_->f               = arr_funcs;
    descr_->metadata        = nullptr;
    descr_->c_metadata      = (NpyAuxData*) new API();
    descr_->hash            = -1;

    if (PyArray_RegisterDataType(descr_) < 0)
      throw py::Exception();

    auto const npy_object = PyArray_DescrFromType(NPY_OBJECT);

    if (PyArray_RegisterCastFunc(
          npy_object, descr_->type_num, 
          (PyArray_VectorUnaryFunc*) cast_from_object) < 0)
      throw py::Exception();
    if (PyArray_RegisterCanCast(
          npy_object, descr_->type_num, NPY_OBJECT_SCALAR) < 0)
      throw py::Exception();
  }

  return descr_;
}


// FIXME: Remove these once Month, Day, Ordinal, Week are 1-indexed.
namespace {

template<class DATE>
inline ora::OrdinalDate
get_ordinal_date_(
  DATE const date)
{
  if (date.is_valid()) 
    return get_ordinal_date(date);
  else
    return ora::OrdinalDate{};
}


template<class DATE>
inline ora::WeekDate
get_week_date_(
  DATE const date)
{
  if (date.is_valid()) 
    return get_week_date(date);
  else
    return ora::WeekDate{};
}


template<class DATE>
inline ora::YmdDate
get_ymd_(
  DATE const date)
{
  if (date.is_valid()) 
    return get_ymd(date);
  else
    return ora::YmdDate{};
}


}  // anonymous namespace


template<class PYDATE>
void
DateDtype<PYDATE>::add(
  Module* const module)
{
  auto const np_module = Module::ImportModule("numpy");

  // Build or get the dtype.
  auto const dtype = DateDtype<PYDATE>::get();
  assert(dtype != nullptr);

  // Add the dtype as a class attribute.
  auto const dict = (Dict*) dtype->typeobj->tp_dict;
  assert(dict != nullptr);
  dict->SetItemString("dtype", (Object*) dtype);

  create_or_get_ufunc(module, "get_day", 1, 1)->add_loop_1(
    dtype->type_num, NPY_UINT8, 
    ufunc_loop_1<Date, npy_bool, ora::date::nex::get_day<Date>>);
  create_or_get_ufunc(module, "get_month", 1, 1)->add_loop_1(
    dtype->type_num, NPY_UINT8, 
    ufunc_loop_1<Date, npy_bool, ora::date::nex::get_month<Date>>);
  create_or_get_ufunc(module, "get_ordinal_date", 1, 1)->add_loop_1(
    dtype, get_ordinal_date_dtype(),
    ufunc_loop_1<Date, ora::OrdinalDate, get_ordinal_date_<Date>>);
  create_or_get_ufunc(module, "get_week_date", 1, 1)->add_loop_1(
    dtype, get_week_date_dtype(),
    ufunc_loop_1<Date, ora::WeekDate, get_week_date_<Date>>);
  create_or_get_ufunc(module, "get_weekday", 1, 1)->add_loop_1(
    dtype->type_num, NPY_UINT8,
    ufunc_loop_1<Date, npy_bool, ora::date::nex::get_weekday<Date>>);
  create_or_get_ufunc(module, "get_year", 1, 1)->add_loop_1(
    dtype->type_num, NPY_INT16, 
    ufunc_loop_1<Date, int16_t, ora::date::nex::get_year<Date>>);
  create_or_get_ufunc(module, "get_ymd", 1, 1)->add_loop_1(
    dtype, get_ymd_dtype(),
    ufunc_loop_1<Date, ora::YmdDate, get_ymd_<Date>>);
  create_or_get_ufunc(module, "get_ymdi", 1, 1)->add_loop_1(
    dtype->type_num, NPY_INT32, 
    ufunc_loop_1<Date, int32_t, ora::date::nex::get_ymdi<Date>>);
  create_or_get_ufunc(module, "is_valid", 1, 1)->add_loop_1(
    dtype->type_num, NPY_BOOL,
    ufunc_loop_1<Date, npy_bool, is_valid>);

  Comparisons<Date, ora::date::nex::equal, ora::date::nex::before>
    ::register_loops(dtype->type_num);

  // Add ufunc loops.
  create_or_get_ufunc(np_module, "add", 2, 1)->add_loop_2(
    dtype->type_num, NPY_INT64, dtype->type_num,
    ufunc_loop_2<Date, int64_t, Date, add>);
  create_or_get_ufunc(np_module, "subtract", 2, 1)->add_loop_2(
    dtype->type_num, NPY_INT64, dtype->type_num,
    ufunc_loop_2<Date, int64_t, Date, subtract_before>);
  create_or_get_ufunc(np_module, "subtract", 2, 1)->add_loop_2(
    dtype->type_num, dtype->type_num, NPY_INT32,
    ufunc_loop_2<Date, Date, int32_t, subtract_between>);
}


//------------------------------------------------------------------------------
// numpy array functions

template<class PYDATE>
void
DateDtype<PYDATE>::copyswap(
  Date* const dst,
  Date const* const src,
  int const swap,
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "copyswap\n";
  if (swap)
    copy_swapped<sizeof(Date)>(src, dst);
  else
    copy<sizeof(Date)>(src, dst);
}


template<class PYDATE>
void 
DateDtype<PYDATE>::copyswapn(
  Date* const dst, 
  npy_intp const dst_stride, 
  Date const* const src, 
  npy_intp const src_stride, 
  npy_intp const n, 
  int const swap, 
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "copyswapn(" << n << ")\n";
  // FIXME: Abstract this out.
  if (src_stride == 0) {
    // Swapper or unswapped fill.  Optimize this special case.
    Date date;
    if (swap) 
      copy_swapped<sizeof(Date)>(src, &date);
    else
      date = *src;
    Date* d = dst;
    for (npy_intp i = 0; i < n; ++i) {
      *d = date;
      d = (Date*) (((char*) d) + dst_stride);
    }
  }
  else {
    char const* s = (char const*) src;
    char* d = (char*) dst;
    if (swap) 
      for (npy_intp i = 0; i < n; ++i) {
        copy_swapped<sizeof(Date)>(s, d);
        s += src_stride;
        d += dst_stride;
      }
    else 
      for (npy_intp i = 0; i < n; ++i) {
        copy<sizeof(Date)>(s, d);
        s += src_stride;
        d += dst_stride;
      }
  }
}


template<class PYDATE>
Object*
DateDtype<PYDATE>::getitem(
  Date const* const data,
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "getitem\n";
  return PYDATE::create(*data).release();
}


template<class PYDATE>
int
DateDtype<PYDATE>::setitem(
  Object* const item,
  Date* const data,
  PyArrayObject* const arr)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "setitem\n";
  try {
    *data = convert_to_date<Date>(item);
  }
  catch (Exception) {
    return -1;
  }
  return 0;
}


template<class PYDATE>
int 
DateDtype<PYDATE>::compare(
  Date const* const d0, 
  Date const* const d1, 
  PyArrayObject* const /* arr */)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "compare\n";
  // Invalid compares least, then missing, then other dates.
  return 
      d0->is_invalid() ? -1
    : d1->is_invalid() ?  1
    : d0->is_missing() ? -1
    : d1->is_missing() ?  1
    : *d0 < *d1        ? -1 
    : *d0 > *d1        ?  1 
    : 0;
}


template<class PYDATE>
void
DateDtype<PYDATE>::cast_from_object(
  Object* const* from,
  Date* to,
  npy_intp num,
  void* /* unused */,
  void* /* unused */)
{
  if (PRINT_ARR_FUNCS)
    std::cerr << "cast_from_object\n";
  for (; num > 0; --num, ++from, ++to) {
    auto const date = maybe_date<Date>(*from);
    *to = date ? *date : Date::INVALID;
  }
}


//------------------------------------------------------------------------------

template<class PYDATE>
ref<Object>
DateDtype<PYDATE>::API::function_date_from_ordinal_date(
  Array* const year_arr,
  Array* const ordinal_arr)
{
  using Date = typename PYDATE::Date;

  // Create the output array.
  auto const size = year_arr->size();
  if (ordinal_arr->size() != size)
    throw py::ValueError("year, ordinal be the same size");
  auto date_arr = Array::SimpleNew1D(size, descr_->type_num);

  // Fill it.
  auto const y = year_arr->get_const_ptr<ora::Year>();
  auto const o = ordinal_arr->get_const_ptr<ora::Ordinal>();
  auto const r = date_arr->get_ptr<Date>();
  for (npy_intp i = 0; i < size; ++i)
    r[i] = ora::date::nex::from_ordinal_date<Date>(y[i], o[i]);

  return std::move(date_arr);
}


template<class PYDATE>
ref<Object>
DateDtype<PYDATE>::API::function_date_from_week_date(
  Array* const week_year_arr,
  Array* const week_arr,
  Array* const weekday_arr)
{
  using Date = typename PYDATE::Date;

  // Create the output array.
  auto const size = week_year_arr->size();
  if (week_arr->size() != size || weekday_arr->size() != size)
    throw py::ValueError("week_year, week, weekday be the same size");
  auto date_arr = Array::SimpleNew1D(size, descr_->type_num);

  // Fill it.
  auto const y = week_year_arr->get_const_ptr<ora::Year>();
  auto const w = week_arr->get_const_ptr<ora::Week>();
  auto const e = weekday_arr->get_const_ptr<ora::Weekday>();
  auto const r = date_arr->get_ptr<Date>();
  for (npy_intp i = 0; i < size; ++i)
    r[i] = ora::date::nex::from_week_date<Date>(y[i], w[i], e[i]);

  return std::move(date_arr);
}


template<class PYDATE>
ref<Object>
DateDtype<PYDATE>::API::function_date_from_ymd(
  Array* const year_arr,
  Array* const month_arr,
  Array* const day_arr)
{
  using Date = typename PYDATE::Date;

  // Create the output array.
  auto const size = year_arr->size();
  if (month_arr->size() != size || day_arr->size() != size)
    throw py::ValueError("year, month, day must be the same size");
  auto date_arr = Array::SimpleNew1D(size, descr_->type_num);

  // Fill it.
  auto const y = year_arr->get_const_ptr<ora::Year>();
  auto const m = month_arr->get_const_ptr<ora::Month>();
  auto const d = day_arr->get_const_ptr<ora::Day>();
  auto const r = date_arr->get_ptr<Date>();
  for (npy_intp i = 0; i < size; ++i)
    r[i] = ora::date::nex::from_ymd<Date>(y[i], m[i], d[i]);

  return std::move(date_arr);
}


template<class PYDATE>
ref<Object>
DateDtype<PYDATE>::API::function_date_from_ymdi(
  Array* const ymdi_arr)
{
  using Date = typename PYDATE::Date;

  // Create the output array.
  auto const size = ymdi_arr->size();
  auto date_arr = Array::SimpleNew1D(size, descr_->type_num);
  // Fill it.
  auto const y = ymdi_arr->get_const_ptr<int>();
  auto const d = date_arr->get_ptr<Date>();
  for (npy_intp i = 0; i < size; ++i)
    d[i] = ora::date::nex::from_ymdi<Date>(y[i]);

  return std::move(date_arr);
}


//------------------------------------------------------------------------------

template<class PYDATE>
PyArray_Descr*
DateDtype<PYDATE>::descr_
  = nullptr;

//------------------------------------------------------------------------------
// Accessories

inline ref<Array>
to_date_array(
  Object* const arg)
{
  if (Array::Check(arg)) {
    // It's an array.  Check its dtype.
    Array* const arr = reinterpret_cast<Array*>(arg);
    if (DateAPI::check(arr->descr()))
      return ref<Array>::of(arr);
  }

  // Convert to an array of the default time dtype.
  auto const def = DateDtype<PyDateDefault>::get();
  return Array::FromAny(arg, def, 0, 0, NPY_ARRAY_BEHAVED);
}


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

