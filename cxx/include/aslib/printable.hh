#pragma once

#include <iostream>
#include <sstream>
#include <string>

// FIXME: Get rid of this whole module.

namespace aslib {

//------------------------------------------------------------------------------

class Printable 
{
public:

  class PrettyPrinter;

  virtual ~Printable() {}

  virtual void print(std::ostream& os) const = 0;
  virtual void _print(std::ostream& os) const { print(os); }
  
  virtual void pretty_print(std::ostream& os) const { print(os); }
  virtual PrettyPrinter pretty() const;

};


class Printable::PrettyPrinter
  : public Printable
{
public:

  PrettyPrinter(Printable const& printable) : printable_(printable) {}
  virtual void print(std::ostream& os) const { printable_.pretty_print(os); }

private:

  Printable const& printable_;

};


inline Printable::PrettyPrinter
Printable::pretty() 
  const
{
  return PrettyPrinter(*this);
}


inline std::ostream& 
operator<<(
  std::ostream& os, 
  const Printable& printable)
{
  printable._print(os);
  return os;
}


//------------------------------------------------------------------------------

template<class C>
class Format
{
public:

  Format(std::string const& pattern, C const& value) : pattern_(pattern), value_(value) {}

  operator std::string() 
    const
  { 
    std::stringstream ss;
    to_stream(ss, pattern_, value_);
    return ss.str();
  }

  void 
  to(std::ostream& os) 
    const 
  { 
    to_stream(os, pattern_, value_);
  }

private:

  std::string const& pattern_;
  C const& value_;

};


template<class C>
inline Format<C>
format(
  std::string const& pattern,
  C const& value)
{
  return Format<C>(pattern, value); 
}


template<class C>
std::ostream&
operator<<(
  std::ostream& os,
  Format<C> const& format)
{
  format.to(os);
  return os;
}


//------------------------------------------------------------------------------

}  // namespace aslib

