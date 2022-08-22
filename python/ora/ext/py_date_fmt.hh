#pragma once

#include <Python.h>

#include "py.hh"
#include "text.hh"

namespace ora {
namespace py {

//------------------------------------------------------------------------------

class PyDateFmt
  : public ExtensionType
{
public:

  static Type type_;
  static void add_to(Module& module);

  PyDateFmt(
    string const& invalid="INVALID",
    string const& missing="MISSING")
  : invalid_(palide(invalid, 10, "", " ", 1, PAD_POS_LEFT_JUSTIFY)),
    missing_(palide(missing, 10, "", " ", 1, PAD_POS_LEFT_JUSTIFY))
  {
  }

  string const invalid_;
  string const missing_;

};


//------------------------------------------------------------------------------

}  // namespace py
}  // namespace ora

