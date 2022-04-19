#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>

#include "ora/date_math.hh"
#include "ora/lib/file.hh"
#include "ora/lib/filename.hh"
#include "ora/posixtz.hh"
#include "ora/time_type.hh"
#include "ora/time_zone.hh"
#include "ora/tzfile.hh"

namespace ora {

using namespace ora::lib;

using std::make_shared;
using std::string;

//------------------------------------------------------------------------------
// Local data
//------------------------------------------------------------------------------

namespace {

std::string
system_time_zone_name;

TimeZone_ptr thread_local
display_time_zone = nullptr;


}  // anonymous namespace


//------------------------------------------------------------------------------

TimeZone::Entry::Entry(
  int64_t _transition,
  TimeZoneOffset offset,
  std::string const& abbrev,
  bool is_dst)
  : transition(_transition)
{
  parts.offset = offset;
  strncpy(
    parts.abbreviation,
    abbrev.c_str(),
    std::min(sizeof(TimeZoneParts::abbreviation), abbrev.length()));
  parts.is_dst = is_dst;
}

TimeZone::Entry::Entry(
  int64_t const transition_time,
  TzFile::Type const& type)
  : transition(transition_time)
{
  parts.offset = type.gmt_offset_;
  parts.is_dst = type.is_dst_;
  // FIXME: This is not future-proof.  Truncate with a warning?
  assert(type.abbreviation_.length() < sizeof(TimeZoneParts::abbreviation));
  strncpy(parts.abbreviation, type.abbreviation_.c_str(), sizeof(TimeZoneParts::abbreviation) - 1);
}


TimeZone::TimeZone()
  : name_("UTC"),
    stop_(std::numeric_limits<int64_t>::max())
{
  entries_.emplace_back(
    time::Unix64Time::MIN.get_offset(),
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
  entries_.emplace_back(time::Unix64Time::MIN.get_offset(), *default_type);

  for (auto const& transition : tz_file.transitions_)
    entries_.emplace_back(
      transition.time_,
      tz_file.types_[transition.type_index_]);
  assert(entries_.size() == tz_file.transitions_.size() + 1);
  std::reverse(begin(entries_), end(entries_));

  // FIXME: For debugging.
  if (tz_file.future_ != "")
    future_ = parse_posix_time_zone(tz_file.future_.c_str());
  if (false && tz_file.future_ != "") {
    std::cerr << "last transition entry: " << entries_.front().transition << "\n";
    std::cerr << "future transitions: " << tz_file.future_ << "\n";
    std::cerr << future_ << "\n";
  }

  stop_ = entries_.front().transition + 1;
}


TimeZoneParts
TimeZone::get_parts(
  int64_t const time)
  const
{
  TimeZone::extend_future(time);

  auto const iter = std::lower_bound(
    entries_.cbegin(), entries_.cend(),
    time,
    [] (Entry const& entry, int64_t const time) { return entry.transition > time; });
  return iter->parts;
}


TimeZoneParts
TimeZone::get_parts_local(
  int64_t const time,
  bool const first)
  const
{
  TimeZone::extend_future(time);

  // First, find the most recent transition, pretending the time is UTC.
  auto const iter = std::lower_bound(
    entries_.cbegin(), entries_.cend(),
    time,
    [] (Entry const& entry, int64_t const time) { return entry.transition > time; });
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
    throw NonexistentDateDaytime();
}


void
TimeZone::extend_future(
  int64_t const until)
  const
{
  if (future_.dst.abbreviation.empty())
    // No future DST.
    return;

  if (until < stop_)
    // Already caught up.
    return;

  if (   future_.start.type != PosixTz::Transition::GREGORIAN
      || future_.end  .type != PosixTz::Transition::GREGORIAN)
    // FIXME: Julian time zone transitions not implemented.  Sorry, Iran.
    return;
  auto const& start = future_.start.spec.gregorian;
  auto const& end   = future_.end  .spec.gregorian;

  // While we're at it, compute a decade of transitions.
  int64_t constexpr DECADE = 10 * 365 * 86400;
  auto stop = (until + DECADE) / DECADE * DECADE;

  std::vector<Entry> entries;
  Datenum const datenum = stop_ / SECS_PER_DAY + DATENUM_UNIX_EPOCH;
  auto year = datenum_to_ordinal_date(datenum).year;
  for (; true; ++year) {
    // Add the DST start transition.
    auto d = weekday_of_month(
      year, start.month,
      start.week == 5 ? -1 : start.week,
      weekday::ENCODING_CLIB::decode(start.weekday));
    auto t =
      (d - DATENUM_UNIX_EPOCH) * SECS_PER_DAY
      + future_.start.ssm
      - future_.std.offset;
    if (stop_ < t)
      entries.emplace_back(t, future_.dst.offset, future_.dst.abbreviation, true);

    // Add the DST stop transition.
    d = weekday_of_month(
      year, end.month,
      end.week == 5 ? -1 : end.week,
      weekday::ENCODING_CLIB::decode(end.weekday));
    t =
      (d - DATENUM_UNIX_EPOCH) * SECS_PER_DAY
      + future_.end.ssm
      - future_.dst.offset;
    if (stop_ < t)
      entries.emplace_back(t, future_.std.offset, future_.std.abbreviation, false);

    if (stop < t)
      break;
  }

  // Append (to front) the new entries.
  std::reverse(entries.begin(), entries.end());
  entries_.insert(entries_.begin(), entries.begin(), entries.end());

  // Note how far we got.
  stop_ = (jan1_datenum(year) - DATENUM_UNIX_EPOCH) * SECS_PER_DAY;
  assert(until <= stop_);
}


TimeZone_ptr
UTC
  = std::make_shared<TimeZone const>();


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

char const* const
ZONEINFO_ENVVAR
  = "ZONEINFO";

fs::Filename const
ZONEINFO_DIR_DEFAULT
  = "/usr/share/zoneinfo";

bool
zoneinfo_dir_initialized
  = false;

fs::Filename
zoneinfo_dir
  {""};

// Cache of loaded time zone objects.
//
// Pointers in this cache should not be
std::map<string, TimeZone_ptr>
time_zones;

}  // anonymous namespace


extern fs::Filename
get_default_zoneinfo_dir()
{
  char const* const env_val = getenv(ZONEINFO_ENVVAR);
  // FIXME: Use the included zoneinfo database.  But where is it installed?
  return env_val != nullptr ? fs::Filename(env_val) : ZONEINFO_DIR_DEFAULT;
}


extern void
set_zoneinfo_dir(
  fs::Filename const& dir)
{
  if (!ora::lib::fs::check(dir, ora::lib::fs::EXISTS, ora::lib::fs::DIRECTORY))
    throw ora::lib::fs::FileNotFoundError(dir);

  // Invalidate cache; contents of the new directory may be different.
  time_zones.clear();
  zoneinfo_dir = dir;
  zoneinfo_dir_initialized = true;
}


extern fs::Filename
get_zoneinfo_dir()
{
  if (! zoneinfo_dir_initialized)
    set_zoneinfo_dir(get_default_zoneinfo_dir());

  return zoneinfo_dir;
}


extern fs::Filename
find_time_zone_file(
  std::string const& name,
  fs::Filename const& zoneinfo_dir)
{
  auto const filename = zoneinfo_dir / name;
  if (check(filename, fs::READ, fs::FILE))
    return filename;
  else
    throw ValueError(std::string("no time zone: ") + name);
}


extern TimeZone_ptr
get_time_zone(
  std::string const& name)
{
  if (name == "UTC" || name == "utc")
    return UTC;

  auto find = time_zones.find(name);
  if (find != end(time_zones))
    return find->second;
  else {
    auto const filename = find_time_zone_file(name);
    return time_zones[name]
      = make_shared<TimeZone const>(TzFile::load(filename), name);
  }
}


extern inline TimeZone
get_time_zone(
  std::string const& name,
  fs::Filename const& zoneinfo_dir)
{
  auto filename = find_time_zone_file(name);
  return TimeZone(TzFile::load(filename), name);
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

      // Look for a link target of the form .../zoneinfo/REGION/TIMEZONE or
      // .../zoneinfo/TIMEZONE, where the prefix is arbitrary.
      fs::Filename const zone_filename = buf;
      auto const parts = get_parts(zone_filename);
      auto const zoneinfo_parts = fs::get_parts(get_zoneinfo_dir());
      if (parts.size() > 1 && parts[1] == "zoneinfo")
        return parts[0];
      else if (parts.size() > 2 && parts[2] == "zoneinfo")
        return parts[1] + '/' + parts[0];
      else
        throw RuntimeError(string("not time zone link: ") + SYSTEM_TIME_ZONE_LINK);
    }
  }
  else
    throw RuntimeError("no system time zone found");
}


extern string
get_system_time_zone_name()
{
  if (system_time_zone_name == "") {
    system_time_zone_name = get_system_time_zone_name_();
    assert(system_time_zone_name != "");
  }
  return system_time_zone_name;
}


extern TimeZone_ptr
get_system_time_zone()
{
  // FIMXE: Store the time zone.
  string const name = get_system_time_zone_name();
  return get_time_zone(name);
}


extern TimeZone_ptr
get_display_time_zone()
{
  if (display_time_zone == nullptr) {
    // Initialize to the value of the TZ environment variable, if set.
    auto const tz_name = getenv("TZ");
    TimeZone_ptr tz;
    if (tz_name == nullptr)
      // TZ is not set; use the system time zone.
      try {
        tz = get_system_time_zone();
      }
      catch (RuntimeError const&) {
        // Unknown system time zone.  Fall back to UTC.
        tz = UTC;
      }
    else
      try {
        tz = get_time_zone(tz_name);
      }
      catch (ValueError const&) {
        // Unknown time zone.  Fall back to UTC.
        tz = UTC;
      }
    set_display_time_zone(std::move(tz));
    assert(display_time_zone != nullptr);
  }
  return display_time_zone;
}


extern void
set_display_time_zone(
  TimeZone_ptr tz)
{
  display_time_zone = tz;
}


//------------------------------------------------------------------------------

}  // namespace ora


