#include <algorithm>
#include <cassert>

#include "filename.hh"

using std::string;

namespace alxs {
namespace fs {

//------------------------------------------------------------------------------

namespace {

inline bool 
is_absolute(
  string const& path)
{
  return path.length() > 0 && path[0] == Filename::SEPARATOR;
}


inline FileType
type_from_stat_mode(
  mode_t mode)
{
  if (S_ISREG(mode))
    return FILE;
  else if (S_ISDIR(mode))
    return DIRECTORY;
  else if (S_ISCHR(mode))
    return CHARACTER_DEVICE;
  else if (S_ISBLK(mode))
    return BLOCK_DEVICE;
  else if (S_ISFIFO(mode))
    return FIFO;
  else if (S_ISLNK(mode))
    return SYMBOLIC_LINK;
  else if (S_ISSOCK(mode))
    return SOCKET;
  // All file system entries should match one of the above.
  assert(false);
}


inline int
access_mode_to_mode(
  AccessMode mode)
{
  switch (mode) {
  case READ:
    return R_OK;
  case WRITE:
    return W_OK;
  case EXECUTE:
    return X_OK;
  case EXISTS:
    return F_OK;
  default:
    assert(false);
  }
}


}  // anonymous namespace

//------------------------------------------------------------------------------

char const 
Filename::SEPARATOR 
  = '/';

char const* const
Filename::DIR_THIS 
  = ".";

char const* const
Filename::DIR_PARENT 
  = "..";

Filename const 
Filename::CURRENT(
  ".");

Filename const 
Filename::ROOT(
  "/");

//------------------------------------------------------------------------------

string 
Filename::normalize(
  string const &pathname)
{
  // Find the first separator.
  string::size_type pos = pathname.find(SEPARATOR);
  // Start with everything up to the first separator.
  string result(pathname, 0, pos);
  while (pos != string::npos) {
    // Find the next separator.
    string::size_type const sep = pathname.find(SEPARATOR, ++pos);
    string const part = pathname.substr(pos, sep - pos);
    if (part == "" || part == DIR_THIS)
      // Empty component (double separator) or same directory; skip.
      ;
    else if (part == DIR_PARENT) {
      // Up one directory, but preserve the .. if it leads a relative path.
      string::size_type const last_sep = result.rfind(SEPARATOR);
      if (last_sep == string::npos && result.length() > 0) {
        result += SEPARATOR;
        result += DIR_PARENT;
      }
      else
        result = result.substr(0, result.rfind(SEPARATOR));
    }
    else {
      // Add a separator to the result.
      result += SEPARATOR;
      // Copy over the next component.
      result += part;
    }
    pos = sep;
  }

  if (fs::is_absolute(pathname) && result == "")
    result = SEPARATOR;

  return result;
}


//------------------------------------------------------------------------------

std::vector<string> 
get_parts(
  Filename const& filename) 
{
  std::vector<string> result;
  for (Filename rest = filename; rest != filename.get_root(); rest = rest.dir())
    result.push_back(rest.base()); 
  return result;
}


Filename 
expand_links(
  Filename const& filename)
{
  auto parts = get_parts(make_absolute(filename));
  std::reverse(parts.begin(), parts.end());

  char buf[PATH_MAX];
  Filename result = Filename::ROOT;
  for (auto part : parts) {
    result /= part;
    ssize_t size = readlink(result, buf, sizeof(buf));
    if (size == -1)
      // Assume not a link (or doesn't exist or inaccessible); continue.
      ;
    else {
      // Use the link target.
      Filename const target(buf);
      result = target.is_absolute() ? target : result.dir() / target;
      // Expand recursively.
      result = expand_links(result);
    }
  }
  return result;
}


struct stat
stat(
  Filename const& filename)
{
  struct stat stat;
  xstat(filename, &stat);
  return stat;
}


struct stat
lstat(
  Filename const& filename)
{
  struct stat stat;
  xlstat(filename, &stat);
  return stat;
}


bool
check(
  Filename const& filename,
  AccessMode mode,
  FileType type)
{
  // Call stat() if we need to check the type.
  if (type != ANY_TYPE) {
    struct stat info;
    int const rval = 
      type == SYMBOLIC_LINK ? lstat(filename, &info)
      : stat(filename, &info);
    if (rval == -1)
      // Can't stat.
      return false;
    if (type != type_from_stat_mode(info.st_mode))
      // Wrong type.
      return false;
  }

  // Call access() if we need to check access, or if we didn't call stat() 
  // and thus still need to check existence.
  if (mode != EXISTS || type == ANY_TYPE) {
    int const rval = access(filename, access_mode_to_mode(mode));
    if (rval != 0)
      return false;
  }

  return true;
}


//------------------------------------------------------------------------------

}  // namespace fs
}  // namespace alxs

