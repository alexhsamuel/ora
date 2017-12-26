#pragma once

#include <cstring>
#include <errno.h>
#include <sstream>
#include <string>

namespace ora {
namespace lib {

//------------------------------------------------------------------------------

class SystemError
  : public std::exception
{
public:

  SystemError(std::string const& call, int errno, std::string const& message)
    : call_(call),
      errno_(errno),
      message_(message)
  {
  }

  SystemError(std::string const& call, std::string const& message)
    : call_(call),
      errno_(errno),
      message_(message)
  {
  }

  SystemError(std::string const& call)
    : call_(call),
      errno_(errno),
      message_("failed")
  {
  }

  virtual ~SystemError() throw () {}

  virtual char const* what() const throw()
  {
    if (what_.size() == 0) 
      what_ = call_ + ": " + message_ + ": " + strerror(errno_);
    return what_.c_str();
  }

private:

  std::string const call_;
  int const errno_;
  std::string const message_;
  std::string mutable what_;

};


//------------------------------------------------------------------------------

class Error
  : public std::exception
{
public:

  Error(std::string const& what) : what_(what) {}
  virtual ~Error() throw () {}
  virtual char const* what() const throw () { return what_.c_str(); }

protected:

  std::string what_;

};


//------------------------------------------------------------------------------

class NameError 
  : public Error
{
public:

  NameError(std::string const& name) : Error(std::string("name error: ") + name) {}
  virtual ~NameError() throw () {}

};


//------------------------------------------------------------------------------

class ValueError 
  : public Error
{
public:

  ValueError(std::string const& name) : Error(std::string("value error: ") + name) {}
  virtual ~ValueError() throw () {}

};


//------------------------------------------------------------------------------

class RangeError
  : public ValueError
{
public:

  RangeError(std::string const& name) : ValueError(std::string("range error: ") + name) {}
  virtual ~RangeError() throw () {}


};


//------------------------------------------------------------------------------

class IndexError
  : public Error
{
public:

  IndexError(size_t index, size_t size) : Error(what_(index, size)) {}
  virtual ~IndexError() throw () {}

protected:

  std::string what_(size_t index, size_t size) const;

};


inline std::string
IndexError::what_(
  size_t index,
  size_t size)
  const
{
  std::ostringstream os;
  os << "IndexError: index " << index << " for size " << size;
  return os.str();
}


//------------------------------------------------------------------------------

class FormatError
  : public Error
{
public:

  FormatError(std::string const& name) : Error(std::string("format error: ") + name) {}
  virtual ~FormatError() throw () {}

};


//------------------------------------------------------------------------------

class RuntimeError
  : public Error
{
public:

  RuntimeError(std::string const& name) : Error(std::string("runtime error: ") + name) {}
  virtual ~RuntimeError() throw () {}

};


//------------------------------------------------------------------------------

}  // namespace lib
}  // namespace ora

