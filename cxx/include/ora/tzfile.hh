#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ora/lib/filename.hh"

namespace ora {

using namespace ora::lib;

//------------------------------------------------------------------------------

class TzFile
{
public:

  struct Type
  {
    int32_t gmt_offset_;
    bool is_dst_;
    std::string abbreviation_;
    bool is_std_;
    bool is_gmt_;
  };

  struct Transition
  {
    int64_t time_;
    uint8_t type_index_;
  };

  struct LeapSeconds
  {
    int64_t time_;
    int32_t duration_;
  };

  static TzFile load(fs::Filename const& filename);

  TzFile() = default;
  TzFile(char const* data, size_t size);
  TzFile(TzFile const&) = default;
  TzFile(TzFile&&) = default;
  TzFile& operator=(TzFile const&) = default;
  TzFile& operator=(TzFile&&) = default;
  ~TzFile() = default;

// FIXME
// private:  

  std::vector<Type> types_;
  std::vector<Transition> transitions_;
  std::vector<LeapSeconds> leap_seconds_;
  std::string future_;

  friend std::ostream& operator<<(std::ostream& os, TzFile const& tz_file);

};


//------------------------------------------------------------------------------

}  // namespace ora


