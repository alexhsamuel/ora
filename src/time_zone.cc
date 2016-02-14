#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <map>

#include "cron/time_zone.hh"
#include "cron/tzfile.hh"
#include "file.hh"
#include "filename.hh"

using std::string;

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------
// Local data
//------------------------------------------------------------------------------

namespace {

bool
system_time_zone_name_initialized
  = false;

std::string
system_time_zone_name;

bool
display_time_zone_initialized
  = false;

TimeZone
display_time_zone;


}  // anonymous namespace


//------------------------------------------------------------------------------

TimeZone::Entry::Entry(
  TimeOffset transition_time,
  TzFile::Type const& type)
  : transition(transition_time)
{
  parts.offset = type.gmt_offset_;
  parts.is_dst = type.is_dst_;
  // FIXME: This is not future-proof.  Truncate with a warning?
  assert(type.abbreviation_.length() < sizeof(TimeZoneParts::abbreviation));
  strncpy(parts.abbreviation, type.abbreviation_.c_str(), sizeof(TimeZoneParts::abbreviation));
  parts.abbreviation[sizeof(TimeZoneParts::abbreviation) - 1] = '\0';
}


TimeZone::TimeZone()
  : name_("UTC")
{
  entries_.emplace_back(
    TIME_OFFSET_MIN, 
    TzFile::Type{0, false, "UTC", true, true});
}


TimeZone::TimeZone(
  TzFile const& tz_file,
  std::string const& name)
  : name_(name)
{
  entries_.reserve(tz_file.transitions_.size() + 1);

  // Find the first non-DST time type.
  assert(tz_file.types_.size() > 0);
  TzFile::Type const* default_type = nullptr;
  for (auto type : tz_file.types_)
    if (! type.is_dst_) {
      default_type = &type;
      break;
    }
  // If we didn't find a non-DST type, use the first type unconditionally.
  if (default_type == nullptr) 
    default_type = &tz_file.types_.front();
  // Add a sentry entry.
  entries_.emplace_back(TIME_OFFSET_MIN, *default_type);

  for (auto const& transition : tz_file.transitions_)
    entries_.emplace_back(
      transition.time_, 
      tz_file.types_[transition.type_index_]);
  assert(entries_.size() == tz_file.transitions_.size() + 1);
  std::reverse(begin(entries_), end(entries_));
}


TimeZoneParts
TimeZone::get_parts(
  TimeOffset time)
  const
{
  auto iter = std::lower_bound(
    entries_.cbegin(), entries_.cend(), 
    time,
    [] (Entry const& entry, TimeOffset time) { return entry.transition > time; });
  return iter->parts;
}


TimeZoneParts
TimeZone::get_parts_local(
  TimeOffset time,
  bool first)
  const
{
  // First, find the most recent transition, pretending the time is UTC.
  auto const iter = std::lower_bound(
    entries_.cbegin(), entries_.cend(), 
    time,
    [] (Entry const& entry, TimeOffset time) { return entry.transition > time; });
  // The sentry protects from no result.
  assert(iter != entries_.cend());

  // We've found the most recent transition for the UTC time, but we want the
  // transition for the local time.  The local time may be before this
  // transition, or after the next.  It's also possible that the local time may
  // have occurred twice (if the local clock was turned back at the transition),
  // or not at all (if the local clock was advanced).
  //
  // We assume that the time zone offset is always smaller than the time between
  // two transitions.  We check whether the local time is actually part of the
  // time zone interval we found, as well as whether it is part of the previous
  // or next intervals, assuming these exist.

  auto const prev = iter + 1;
  auto const next = iter - 1;

  bool const in_prev
    = prev != entries_.cend()
      && prev->transition + prev->parts.offset <= time
      && time < (prev - 1)->transition + prev->parts.offset;
  bool const in_this
    = iter->transition + iter->parts.offset <= time
      && (iter == entries_.cbegin()
          || time < (iter - 1)->transition + iter->parts.offset);
  bool const in_next
    = iter != entries_.cbegin()
      && next->transition + next->parts.offset <= time
      && (next == entries_.cbegin()
          || time < (next - 1)->transition + next->parts.offset);

  // FIXME: For debugging.
  if (false) {
    std::cerr << "offset for local " << time << '\n';
    std::cerr 
      << "prev? " << (iter + 1)->transition 
      << "/" << (iter + 1)->parts.offset
      << " = " << (iter + 1)->transition + (iter + 1)->parts.offset
      << "-" << (iter + 0)->transition + (iter + 1)->parts.offset
      << " -> " << (in_prev ? "true" : "false") << '\n';
    std::cerr 
      << "this? " << (iter + 0)->transition 
      << "/" << (iter + 0)->parts.offset
      << " = " << (iter + 0)->transition + (iter + 0)->parts.offset
      << "-" << (iter - 1)->transition + (iter + 0)->parts.offset
      << " -> " << (in_this ? "true" : "false") << '\n';
    std::cerr 
      << "next? " << (iter - 1)->transition 
      << "/" << (iter - 1)->parts.offset
      << " = " << (iter - 1)->transition + (iter - 1)->parts.offset
      << "-" << (iter - 2)->transition + (iter - 1)->parts.offset
      << " -> " << (in_next ? "true" : "false") << '\n';
  }

  if (in_this)
    // The local time is part of the transition interval we found, but if it
    // occurred in the previous or next as well, we need to disambiguate.
    return 
        in_prev ? (first ? (iter + 1)->parts : iter->parts)
      : in_next ? (first ? iter->parts : (iter - 1)->parts)
      : iter->parts;
  else if (in_prev)
    // Actually, it's only in the previous transition interval.
    return (iter + 1)->parts;
  else if (in_next)
    // Actually, it's only in the next transition interval.
    return (iter - 1)->parts;
  else
    // The local time does not exist.
    throw NonexistentLocalTime(); 
}


TimeZone const
UTC;


//------------------------------------------------------------------------------
// Functions.
//------------------------------------------------------------------------------

namespace {

fs::Filename const
SYSTEM_TIME_ZONE_FILE
  = "/etc/timezone";

fs::Filename const
SYSTEM_TIME_ZONE_LINK
  = "/etc/localtime";

fs::Filename const
ZONEINFO_DIR 
  = "/usr/share/zoneinfo";

std::map<std::string, TimeZone>
time_zones;

}  // anonymous namespace


extern fs::Filename
find_time_zone_file(
  std::string const& name)
{
  // FIXME.
  // auto const filename = ZONEINFO_DIR / name;
  auto const filename = fs::Filename("/home/samuel/sw/tzdata2016a/etc/zoneinfo") / name;
  if (check(filename, fs::READ, fs::FILE))
    return filename;
  else
    throw ValueError(std::string("no time zone: ") + name);
}


extern TimeZone const&
get_time_zone(
  std::string const& name,
  bool reload)
{
  auto find = time_zones.find(name);
  if (reload && find != end(time_zones))
    time_zones.erase(find);
  if (reload || find == end(time_zones)) {
    auto filename = find_time_zone_file(name);
    return time_zones[name] = TimeZone(TzFile::load(filename), name);
  }
  else
    return find->second;
}


string
get_system_time_zone_name_()
{
  // FIXME: Portability.
  if (fs::check(SYSTEM_TIME_ZONE_FILE, fs::READ, fs::FILE)) {
    std::ifstream file(SYSTEM_TIME_ZONE_FILE.operator string());
    char line[128];
    file.getline(line, sizeof(line));
    line[sizeof(line) - 1] = '\0';
    string time_zone = strip(string(line));
    if (time_zone.length() > 0)
      return time_zone;
    else
      throw RuntimeError(
        string("no time zone name in ") + SYSTEM_TIME_ZONE_FILE);
  }
  else if (fs::check(SYSTEM_TIME_ZONE_LINK, fs::EXISTS, fs::SYMBOLIC_LINK)) {
    char buf[PATH_MAX];
    int const result = 
      readlink(SYSTEM_TIME_ZONE_LINK, buf, sizeof(buf));
    if (result == -1) 
      throw RuntimeError(string("can't read link: ") + SYSTEM_TIME_ZONE_LINK);
    else {
      // Nul-terminate.
      assert(result < PATH_MAX);
      buf[result] = '\0';

      fs::Filename const zone_filename = buf;
      auto const parts = get_parts(zone_filename);
      auto const zoneinfo_parts = fs::get_parts(ZONEINFO_DIR);
      if (parts.size() == zoneinfo_parts.size() + 2)
        return parts[1] + '/' + parts[0];
      else
        throw RuntimeError(string("not time zone link: ") + SYSTEM_TIME_ZONE_LINK);
    }
  }
  else 
    throw RuntimeError("no system time zone found");
}


extern string
get_system_time_zone_name(
  bool reload)
{
  if (reload || ! system_time_zone_name_initialized) {
    system_time_zone_name = get_system_time_zone_name_();
    system_time_zone_name_initialized = true;
  }
  return system_time_zone_name;
}


extern TimeZone const&
get_system_time_zone(
  bool reload)
{
  // FIXME: Portability.
  string const name = get_system_time_zone_name(reload);
  return get_time_zone(name, reload);
}


extern TimeZone const&
get_display_time_zone()
{
  if (! display_time_zone_initialized) {
    set_display_time_zone(get_system_time_zone());
    assert(display_time_zone_initialized);
  }
  return display_time_zone;
}


extern void
set_display_time_zone(
  TimeZone const& tz)
{
  display_time_zone = tz;
  display_time_zone_initialized = true;
}


//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs

