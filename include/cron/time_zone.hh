#pragma once

#include <cstdint>
#include <vector>

#include "cron/types.hh"
#include "cron/tzfile.hh"
#include "string.hh"

namespace alxs {
namespace cron {

//------------------------------------------------------------------------------

class TimeZone
{
public:

  TimeZone();
  TimeZone(TimeZone const&) = default;
  TimeZone(TimeZone&&) = default;
  TimeZone(TzFile const& tz_file, std::string const& name);
  TimeZone& operator=(TimeZone const&) = default;
  TimeZone& operator=(TimeZone&&) = default;

  std::string get_name() const { return name_; }

  TimeZoneParts get_parts(TimeOffset time) const;

  template<class TIME> 
  TimeZoneParts 
  get_parts(
    TIME time) 
    const
  {
    return get_parts(time.get_time_offset());
  }

  TimeZoneParts get_parts_local(TimeOffset, bool first=true) const;

  TimeZoneParts get_parts_local(
    Datenum datenum, 
    Daytick daytick, 
    bool first=true) 
    const
   {
     return get_parts_local(
       (datenum - DATENUM_UNIX_EPOCH) * SECS_PER_DAY 
         + (TimeOffset) (daytick / DAYTICK_PER_SEC),
       first);
   }

private:

  struct Entry
  {
    Entry(TimeOffset transition, TzFile::Type const& type);

    TimeOffset transition;
    TimeZoneParts parts;
  };

  std::string name_;
  std::vector<Entry> entries_;

};


extern TimeZone const UTC;

extern fs::Filename get_zoneinfo_dir();
extern void set_zoneinfo_dir(fs::Filename const& dir);
extern fs::Filename find_time_zone_file(std::string const& name);
extern TimeZone const& get_time_zone(std::string const& name, bool reload=false);

extern std::string get_system_time_zone_name(bool reload=false);
extern TimeZone const& get_system_time_zone(bool reload=false);

extern TimeZone const& get_display_time_zone();
extern void set_display_time_zone(TimeZone const& tz);

inline void 
set_display_time_zone(
  std::string const& name)
{
  set_display_time_zone(get_time_zone(name));
}


//------------------------------------------------------------------------------

}  // namespace cron
}  // namespace alxs

