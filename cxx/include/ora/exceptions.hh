#pragma once

#include <string>

#include "ora/types.hh"

namespace ora {

using namespace std::string_literals;

//------------------------------------------------------------------------------
// Exceptions
//------------------------------------------------------------------------------

class CalendarError
: public Error
{
public:

  using Error::Error;
  virtual ~CalendarError() = default;

};


class CalendarRangeError
: public CalendarError
{
public:

  CalendarRangeError() : CalendarError("date not in calendar range") {}
  virtual ~CalendarRangeError() = default;

};


class DateError
  : public Error
{
public:

  using Error::Error;
  virtual ~DateError() = default;

};


class DateFormatError
: public DateError
{
public:

  using DateError::DateError;
  virtual ~DateFormatError() = default;

};


class DateRangeError
  : public DateError
{
public:

  DateRangeError() : DateError("date not in range") {}
  virtual ~DateRangeError() = default;

};


class InvalidDateError
  : public DateError
{
public:

  InvalidDateError() : DateError("invalid date") {}
  virtual ~InvalidDateError() = default;

};


class DaytimeError
  : public Error
{
public:

  using Error::Error;
  virtual ~DaytimeError() = default;

};


class DaytimeRangeError
  : public DaytimeError
{
public:

  DaytimeRangeError() : DaytimeError("daytime not in range") {}
  virtual ~DaytimeRangeError() = default;

};


class DaytimeFormatError
  : public DaytimeError
{
public:

  using DaytimeError::DaytimeError;
  virtual ~DaytimeFormatError() = default;

};


class TimeError
  : public Error
{
public:

  using Error::Error;
  virtual ~TimeError() = default;

};


class InvalidTimeError
  : public TimeError
{
public:

  InvalidTimeError() : TimeError("invalid time") {}
  virtual ~InvalidTimeError() = default;

};


class TimeRangeError
  : public TimeError
{
public:

  TimeRangeError() : TimeError("time out of range") {}
  virtual ~TimeRangeError() noexcept = default;

};


class InvalidDaytimeError
  : public DaytimeError
{
public:

  InvalidDaytimeError() : DaytimeError("invalid daytime") {}
  virtual ~InvalidDaytimeError() = default;

};


class NonexistentDateDaytime
  : public Error
{
public:

  NonexistentDateDaytime() : Error("local time does not exist") {}
  virtual ~NonexistentDateDaytime() = default;

};


class TimeFormatError
  : public FormatError
{
public:

  TimeFormatError(std::string const& name) : FormatError(std::string("in time pattern: ") + name) {}
  virtual ~TimeFormatError() = default;

};


class TimeParseError
  : public TimeError
{
public:

  TimeParseError(char const* const string): TimeError("can't parse time: "s + string) {}
  virtual ~TimeParseError() = default;

};


class UnknownTimeZoneError
  : public Error
{
public:

  UnknownTimeZoneError(std::string const& name) : Error(std::string("unknown time zone: " + name)) {}
  virtual ~UnknownTimeZoneError() = default;

};


//------------------------------------------------------------------------------

}  // namespace ora

