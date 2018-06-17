#pragma once

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>

#include "np.hh"
#include "py.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

namespace np {

int constexpr SECOND_TYPE           = NPY_FLOAT64;
int constexpr MINUTE_TYPE           = NPY_UINT8;
int constexpr HOUR_TYPE             = NPY_UINT8;
int constexpr DAY_TYPE              = NPY_UINT8;
int constexpr MONTH_TYPE            = NPY_UINT8;
int constexpr YEAR_TYPE             = NPY_INT16;
int constexpr ORDINAL_TYPE          = NPY_UINT16;
int constexpr WEEK_TYPE             = NPY_UINT8;
int constexpr WEEKDAY_TYPE          = NPY_UINT8;
int constexpr DAYTICK_TYPE          = NPY_UINT64;
int constexpr DATENUM_TYPE          = NPY_UINT32;
int constexpr YMDI_TYPE             = NPY_INT32;
int constexpr SSM_TYPE              = NPY_FLOAT64;
int constexpr TIME_ZONE_OFFSET_TYPE = NPY_INT32;
int constexpr TIME_OFFSET_TYPE      = NPY_INT64;

}  // namespace np

//------------------------------------------------------------------------------

extern PyArray_Descr* get_ordinal_date_dtype();
extern PyArray_Descr* get_week_date_dtype();
extern PyArray_Descr* get_ymd_dtype();

//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

