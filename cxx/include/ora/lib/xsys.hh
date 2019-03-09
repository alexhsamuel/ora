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

#include "ora/lib/exc.hh"

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
    throw ora::lib::SystemError("close");
}


inline int xdup(int fd)
{
  int const dup_fd = dup(fd);
  if (dup_fd == -1)
    throw ora::lib::SystemError("dup");
  return dup_fd;
}


inline int xdup2(int old_fd, int new_fd)
{
  int const fd = dup2(old_fd, new_fd);
  if (fd != new_fd)
    throw ora::lib::SystemError("dup2");
  return fd;
}


inline void xexecv(char const* filename, char* const argv[])
{
  execv(filename, argv);
  throw ora::lib::SystemError("execv");
}


inline void xexecve(char const* filename, char* const argv[], char* const envp[])
{
  execve(filename, argv, envp);
  throw ora::lib::SystemError("execve");
}


inline pid_t xfork()
{
  pid_t const pid = fork();
  if (pid == -1)
    throw ora::lib::SystemError("fork");
  return pid;
}


inline void xfstat(int fd, struct stat* buf)
{
  int const rval = fstat(fd, buf);
  if (rval == -1)
    throw ora::lib::SystemError("fstat");
}


inline char* xgetcwd(char* buf, size_t size)
{
  char* const cwd = getcwd(buf, size);
  if (cwd == NULL)
    throw ora::lib::SystemError("getcwd");
  return cwd;
}


inline void xgettimeofday(struct timeval* tv, struct timezone* tz=nullptr)
{
  int const rval = gettimeofday(tv, tz);
  if (rval == -1)
    throw ora::lib::SystemError("gettimeofday");
}


inline off_t xlseek(int fd, off_t offset, int whence)
{
  off_t const off = lseek(fd, offset, whence);
  if (off == -1)
    throw ora::lib::SystemError("lseek");
  return off;
}


inline void xlstat(
  char const* path,
  struct stat* buf)
{
  int const rval = lstat(path, buf);
  if (rval == -1)
    throw ora::lib::SystemError("lstat");
}


inline int xmkstemp(char* name_template)
{
  int const fd = mkstemp(name_template);
  if (fd == -1)
    throw ora::lib::SystemError("mkstemp");
  return fd;
}


inline int xopen(const char* pathname, int flags, mode_t mode=0666)
{
  int const fd = open(pathname, flags, mode);
  if (fd == -1)
    throw ora::lib::SystemError("open");
  return fd;
}


inline size_t xread(int fd, void* buf, size_t count)
{
  ssize_t const rval = read(fd, buf, count);
  if (rval == -1)
    // FIXME: Handle EINTR here?
    throw ora::lib::SystemError("read");
  return (size_t) rval;
}


inline char* xrealpath(char const* path, char* resolved_path)
{
  char* const new_path = realpath(path, resolved_path);
  if (new_path == NULL)
    throw ora::lib::SystemError("realpath");
  return new_path;
}


inline void xstat(
  char const* path,
  struct stat* buf)
{
  int const rval = stat(path, buf);
  if (rval == -1)
    throw ora::lib::SystemError("stat");
}


inline void xunlink(char const* pathname)
{
  int const rval = unlink(pathname);
  if (rval == -1)
    throw ora::lib::SystemError("unlink");
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
    throw ora::lib::SystemError("wait4");
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
    throw ora::lib::SystemError("waitid");
}


//------------------------------------------------------------------------------



