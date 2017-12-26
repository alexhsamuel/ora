#pragma once

#include <cassert>
#include <cstdlib>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "aslib/exc.hh"

//------------------------------------------------------------------------------

// If NDEBUG, assert(x) is #define'd as (void)(0).  If 'x' references a variable
// that is otherwise not used, this may result in an unused variable warning.
// Redefine the NDEBUG assert() to suppress this.

// FIXME: Bad idea!!  Add check_*() functions instead.
#ifdef NDEBUG
# undef assert
# define assert(e) do { (void) (e); } while (false)
#endif

//------------------------------------------------------------------------------

void 
xexecv(
  char const* filename, 
  char* const argv[])
  __attribute__ ((__noreturn__));

void 
xexecve(
  char const* filename, 
  char* const argv[], 
  char* const envp[])
  __attribute__ ((__noreturn__));


//------------------------------------------------------------------------------

inline void xclose(int fd)
{
  int const rval = close(fd);
  if (rval == -1)
    throw aslib::SystemError("close");
  assert(rval == 0);
}


inline int xdup(int fd)
{
  int const dup_fd = dup(fd);
  if (dup_fd == -1)
    throw aslib::SystemError("dup");
  return dup_fd;
}


inline int xdup2(int old_fd, int new_fd)
{
  int const fd = dup2(old_fd, new_fd);
  if (fd != new_fd)
    throw aslib::SystemError("dup2");
  return fd;
}


inline void xexecv(char const* filename, char* const argv[])
{
  int const rval = execv(filename, argv);
  assert(rval == -1);
  throw aslib::SystemError("execv");
}


inline void xexecve(char const* filename, char* const argv[], char* const envp[])
{
  int const rval = execve(filename, argv, envp);
  assert(rval == -1);
  throw aslib::SystemError("execve");
}


inline pid_t xfork()
{
  pid_t const pid = fork();
  if (pid == -1)
    throw aslib::SystemError("fork");
  return pid;
}


inline void xfstat(int fd, struct stat* buf)
{
  int const rval = fstat(fd, buf);
  if (rval == -1)
    throw aslib::SystemError("fstat");
  assert(rval == 0);
}


inline char* xgetcwd(char* buf, size_t size)
{
  char* const cwd = getcwd(buf, size);
  if (cwd == NULL)
    throw aslib::SystemError("getcwd");
  assert(cwd == buf);
  return cwd;
}


inline void xgettimeofday(struct timeval* tv, struct timezone* tz=nullptr)
{
  int const rval = gettimeofday(tv, tz);
  if (rval != 0) {
    assert(rval == -1);
    throw aslib::SystemError("gettimeofday");
  }
}


inline off_t xlseek(int fd, off_t offset, int whence)
{
  off_t const off = lseek(fd, offset, whence);
  if (off == -1)
    throw aslib::SystemError("lseek");
  return off;
}


inline void xlstat(
  char const* path,
  struct stat* buf)
{
  int const rval = lstat(path, buf);
  if (rval == -1)
    throw aslib::SystemError("lstat");
  assert(rval == 0);
}


inline int xmkstemp(char* name_template)
{
  int const fd = mkstemp(name_template);
  if (fd == -1)
    throw aslib::SystemError("mkstemp");
  assert(fd >= 0);
  return fd;
}


inline int xopen(const char* pathname, int flags, mode_t mode=0666)
{
  int const fd = open(pathname, flags, mode);
  if (fd == -1)
    throw aslib::SystemError("open");
  return fd;
}


inline size_t xread(int fd, void* buf, size_t count)
{
  ssize_t const rval = read(fd, buf, count);
  if (rval == -1)
    // FIXME: Handle EINTR here?
    throw aslib::SystemError("read");
  return (size_t) rval;
}


inline char* xrealpath(char const* path, char* resolved_path)
{
  char* const new_path = realpath(path, resolved_path);
  if (new_path == NULL)
    throw aslib::SystemError("realpath");
  assert(resolved_path == NULL || new_path == resolved_path);
  return new_path;
}


inline void xstat(
  char const* path,
  struct stat* buf)
{
  int const rval = stat(path, buf);
  if (rval == -1)
    throw aslib::SystemError("stat");
  assert(rval == 0);
}


inline void xunlink(char const* pathname)
{
  int const rval = unlink(pathname);
  if (rval == -1)
    throw aslib::SystemError("unlink");
  assert(rval == 0);
}


inline pid_t
xwait4(
  pid_t pid, 
  int* status, 
  int options, 
  struct rusage* usage=NULL)
{
  pid_t const rval = wait4(pid, status, options, usage);
  if (rval == -1)
    throw aslib::SystemError("wait4");
  if (pid > 0)
    assert(rval == pid);
  return rval;
}


inline void
xwaitid(
  idtype_t idtype,
  id_t id,
  siginfo_t* infop,
  int options)
{
  int const rval = waitid(idtype, id, infop, options);
  if (rval != 0)
    throw aslib::SystemError("waitid");
}


//------------------------------------------------------------------------------



