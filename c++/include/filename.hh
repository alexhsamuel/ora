#ifndef __FILENAME_HH__
#define __FILENAME_HH__

#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
#include <libgen.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "exc.hh"
#include "xsys.hh"

namespace alxs {
namespace fs {

//------------------------------------------------------------------------------

enum FileType 
{
  ANY_TYPE,
  SOCKET,
  SYMBOLIC_LINK,
  FILE,
  BLOCK_DEVICE,
  DIRECTORY,
  CHARACTER_DEVICE,
  FIFO,
};

enum AccessMode
{
  READ,
  WRITE,
  EXECUTE,
  EXISTS
};

//------------------------------------------------------------------------------

class Filename
{
public:

  static char const SEPARATOR;
  static char const* const DIR_THIS;
  static char const* const DIR_PARENT;

  static Filename const ROOT;
  static Filename const CURRENT;

  static std::string normalize(std::string const& pathname);

  Filename(Filename const& filename);
  Filename(std::string const& pathname);
  Filename(char const* pathname);
  Filename const& operator=(Filename const& filename);
  Filename const& operator=(std::string const& pathname);
  ~Filename();

  bool operator==(Filename const& filename) const;
  bool operator!=(Filename const& filename) const;

  static Filename get_cwd();

  operator std::string() const { return pathname_; }
  std::string as_string() const { return pathname_; }
  operator char const*() const { return pathname_.c_str(); }
  char const* as_c_str() const { return pathname_.c_str(); }

  bool is_absolute() const  { return pathname_.length() > 0 && pathname_[0] == SEPARATOR; }
  Filename get_root() const { return is_absolute() ? ROOT : CURRENT; }

  Filename dir() const;
  std::string base() const;

  Filename operator/(std::string const& base) const;
  Filename const& operator/=(std::string const& base);

protected:

  Filename(std::string const& pathname, bool normalize);

private:

  std::string pathname_;

};


inline Filename::Filename(Filename const& filename)
  : pathname_(filename.pathname_)
{
}


inline Filename::Filename(std::string const& pathname)
  : pathname_(normalize(pathname))
{
}


inline Filename::Filename(char const* pathname)
  : pathname_(normalize(pathname))
{
}


inline Filename const& Filename::operator=(Filename const& filename)
{
  pathname_ = filename.pathname_;
  return *this;
}


inline Filename const& Filename::operator=(std::string const& pathname)
{
  pathname_ = normalize(pathname);
  return *this;
}


inline Filename::~Filename()
{
}


inline bool Filename::operator==(Filename const& filename) const
{
  // FIXME: Is this the beavior we want?
  return filename.pathname_ == pathname_;
}


inline bool Filename::operator!=(Filename const& filename) const
{
  return ! operator==(filename);
}


inline Filename Filename::get_cwd()
{
  char pathname[PATH_MAX];
  char* const cwd = xgetcwd(pathname, sizeof(pathname));
  return Filename(cwd, false);
}


inline Filename Filename::dir() const
{
  char pathname[pathname_.size() + 1];
  strcpy(pathname, pathname_.c_str());
  char* dir = dirname(pathname);
  return Filename(dir);
}


inline std::string Filename::base() const
{
  char pathname[pathname_.size() + 1];
  strcpy(pathname, pathname_.c_str());
  char* base = basename(pathname);
  return std::string(base);
}


inline Filename Filename::operator/(std::string const& base) const
{
  return Filename(pathname_ + SEPARATOR + base);
}


inline Filename const& Filename::operator/=(std::string const& base) 
{
  pathname_ = normalize(pathname_ + SEPARATOR + base);
  return *this;
}


inline Filename::Filename(std::string const& pathname, bool normalize)
  : pathname_(normalize ? Filename::normalize(pathname) : pathname)
{
}


//------------------------------------------------------------------------------
// Other overloads
//------------------------------------------------------------------------------

inline
std::string
operator+(
  std::string const& str,
  Filename const& fn)
{
  return str + fn.as_string();
}


//------------------------------------------------------------------------------

inline Filename make_absolute(Filename const& filename)
{
  if (filename.is_absolute())
    return filename;
  else 
    return Filename::get_cwd() / filename;
}


inline Filename make_absolute(Filename const& filename, Filename const& cwd)
{
  if (filename.is_absolute())
    return filename;
  else {
    assert(cwd.is_absolute());
    return cwd / filename;
  }
}


//------------------------------------------------------------------------------

extern std::vector<std::string> get_parts(Filename const& filename);

/**
 * Expands symbolic links.  Links are read on a best-effort basis; if a pathname
 * component is inaccessible, it is left unmodified.
 */
extern Filename expand_links(Filename const& filename);

extern struct stat stat(Filename const& filename);
extern bool check(Filename const& filename, AccessMode mode=EXISTS, FileType type=ANY_TYPE);

//------------------------------------------------------------------------------

}  // namespace fs
}  // namespace alxs

#endif  // #ifndef __FILENAME_HH__

