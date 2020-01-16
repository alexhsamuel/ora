// FIXME
#if 0
#include <endian.h>
#else
#define be32toh __builtin_bswap32
#define be64toh __builtin_bswap64
#endif

#include <iomanip>
#include <iostream>

#include "ora/lib/exc.hh"
#include "ora/lib/file.hh"
#include "ora/lib/filename.hh"
#include "ora/tzfile.hh"

//------------------------------------------------------------------------------

namespace {

using ora::lib::FormatError;
using std::string;

class Scanner
{
public:

  Scanner(char const* data, size_t size) : pos_(data), end_(data + size) {}

  bool is_empty() const { return pos_ == end_; }
  char const* get_position() const { return pos_; }
  void skip(size_t size);
  template<class T> T next();

private:

  template<class T> static T swap(T value);

  char const* pos_;
  char const* const end_;

};


inline void
Scanner::skip(
  size_t size)
{
  if (end_ < pos_ + size)
    throw FormatError("unexpected end of data");
  pos_ += size;
}


template<class T>
inline T
Scanner::next()
{
  if (end_ < pos_ + sizeof(T))
    throw FormatError("unexpected end of data");
  T const value = *reinterpret_cast<T const*>(pos_);
  pos_ += sizeof(T);
  return swap(value);
}


template<> inline char     Scanner::swap<char    >(char     val) { return val; }
template<> inline uint8_t  Scanner::swap<uint8_t >(uint8_t  val) { return val; }
template<> inline int8_t   Scanner::swap<int8_t  >(int8_t   val) { return val; }
template<> inline uint32_t Scanner::swap<uint32_t>(uint32_t val) { return be32toh(val); }
template<> inline int32_t  Scanner::swap<int32_t >(int32_t  val) { return be32toh(val); }
template<> inline int64_t  Scanner::swap<int64_t >(int64_t  val) { return be64toh(val); }


string
to_asctime(
  time_t time)
{
  struct tm tm;
  gmtime_r(&time, &tm);
  char time_str[26] = "??? ??? ?? ??:??:?? ????";
  asctime_r(&tm, time_str);
  return string(time_str, 24);
}


void
check_header(
  Scanner& scanner)
{
  if (   scanner.next<char>() != 'T'
      || scanner.next<char>() != 'Z'
      || scanner.next<char>() != 'i'
      || scanner.next<char>() != 'f')
    throw FormatError("not a tz file");
  if (scanner.next<char>() != '2')
    throw FormatError("not a tz file version 2");
  for (size_t i = 0; i < 15; ++i)
    if (scanner.next<char>() != 0)
      throw FormatError("tz file wrong padding");
}


}  // anonymous namespace


//------------------------------------------------------------------------------

namespace ora {

using namespace ora::lib;


TzFile
TzFile::load(
  fs::Filename const& filename)
{
  string const data = load_text(filename);
  return TzFile(data.c_str(), data.length());
}


TzFile::TzFile(
  char const* data,
  size_t size)
{
  Scanner scanner(data, size);

  // Check the initial header.
  check_header(scanner);
  // Skip the version-0 data, which uses 32-bit epoch times.
  {
    uint32_t const ttisgmtcnt = scanner.next<uint32_t>();
    uint32_t const ttisstdcnt = scanner.next<uint32_t>();
    uint32_t const leapcnt    = scanner.next<uint32_t>();
    uint32_t const timecnt    = scanner.next<uint32_t>();
    uint32_t const typecnt    = scanner.next<uint32_t>();
    uint32_t const charcnt    = scanner.next<uint32_t>();
    scanner.skip(
            ttisgmtcnt 
      +     ttisstdcnt 
      + 8 * leapcnt 
      + 5 * timecnt 
      + 6 * typecnt 
      +     charcnt
      );
  }

  // Check the second header.
  check_header(scanner);

  // Get item counts.
  uint32_t const ttisgmtcnt = scanner.next<uint32_t>();
  uint32_t const ttisstdcnt = scanner.next<uint32_t>();
  uint32_t const leapcnt    = scanner.next<uint32_t>();
  uint32_t const timecnt    = scanner.next<uint32_t>();
  uint32_t const typecnt    = scanner.next<uint32_t>();
  uint32_t const charcnt    = scanner.next<uint32_t>();

  // Make room.
  transitions_.resize(timecnt);

  // Get local time transitions.
  for (size_t i = 0; i < timecnt; ++i)
    transitions_[i].time_ = scanner.next<int64_t>();
  // Get local time types corresponding to transitions.
  for (size_t i = 0; i < timecnt; ++i) {
    uint8_t const type_index = scanner.next<uint8_t>();
    if (typecnt <= type_index)
      throw FormatError("invalid local time type for transition");
    transitions_[i].type_index_ = type_index;
  }

  // Compute the start of the abbreviation character block.
  char const* abbreviations = scanner.get_position() +  6 * typecnt;
  // Get local time types.
  types_.resize(typecnt);
  for (size_t i = 0; i < typecnt; ++i) {
    Type& type = types_[i];
    type.gmt_offset_ = scanner.next<int32_t>();
    type.is_dst_ = scanner.next<int8_t>() != 0;
    type.abbreviation_ = string(abbreviations + scanner.next<uint8_t>());
  }

  // Skip over the abbreviation block; we've already used it.
  scanner.skip(charcnt);

  // Get leap seconds.
  leap_seconds_.resize(leapcnt);
  for (size_t i = 0; i < leapcnt; ++i) {
    LeapSeconds& leap = leap_seconds_[i];
    leap.time_ = scanner.next<int64_t>();
    leap.duration_ = scanner.next<int32_t>();
  }

  // Get is-standard and is-GMT flags.
  if (ttisstdcnt > typecnt)
    throw FormatError("invalid tzh_ttisstdcnt");
  for (size_t i = 0; i < ttisstdcnt; ++i)
    types_[i].is_std_ = scanner.next<int8_t>() != 0;
  if (ttisgmtcnt > typecnt)
    throw FormatError("invalid tzh_ttisgmtcnt");
  for (size_t i = 0; i < ttisgmtcnt; ++i)
    types_[i].is_gmt_ = scanner.next<int8_t>() != 0;

  // Get the indication for additional future transitions.
  if (scanner.next<char>() != '\n')
    throw FormatError("expected newline before POSIX TZ string");
  while (true) {
    char const c = scanner.next<char>();
    if (c == '\n')
      break;
    else
      future_ += c;
  }
  
  // Should be nothing left.
  if (! scanner.is_empty())
    throw FormatError("unexpected additional data");
}


std::ostream& 
operator<<(
  std::ostream& os, 
  TzFile const& tz_file)
{
  os << "Time zone file:\n"
     << "  types (" << tz_file.types_.size() << ")\n"
     << "  transitions:\n";
  for (auto const& trans : tz_file.transitions_) {
    TzFile::Type const& type = tz_file.types_[trans.type_index_];
    os << "    time:";
    if (trans.time_ == -576460752303423488)
      os << "         min";
    else
      os << std::setw(12) << trans.time_;
    os
       << " = " << to_asctime((time_t) trans.time_)
       << " to '" << type.abbreviation_
       << "' offset:" << type.gmt_offset_
       << " sec DST:" << (type.is_dst_ ? "Y" : "N")
       << " std:" << (type.is_std_ ? "Y" : "N")
       << " GMT:" << (type.is_gmt_ ? "Y" : "N")
       << "\n";
  }
  os << "  leap seconds (" << tz_file.leap_seconds_.size() << "):\n";
  for (auto const& leap : tz_file.leap_seconds_) 
    os << "    time:" << to_asctime((time_t) leap.time_)
       << " duration:" << leap.duration_
       << " secs\n";
  os << "  future transitions: '" << tz_file.future_
     << "'\n"
     << "\n";

  return os;
}


//------------------------------------------------------------------------------

}  // namespace ora


