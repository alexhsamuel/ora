import os
from   pathlib import Path
from   typing import Iterable

from   .ext import Date
from   .ext import make_weekday_calendar, parse_calendar, make_const_calendar
from   .weekday import parse_weekdays

#-------------------------------------------------------------------------------

def load_calendar_file(path, *, name=None):
    """
    Loads a calendar from the file at `path`.

    The file has the following format::

        START date
        STOP date
        date
        date
        ...

    Each 'date' is in YYYY-MM-DD format.  Blank lines are ignored.  Text on
    each line following the date is ignored.

    :param name:
      The calendar name.  If `None`, the file's stem name is used.
    """
    path = Path(path)
    with open(path, "r") as file:
        cal = parse_calendar(file)
    cal.name = path.stem if name is None else name
    return cal


def load_business_calendar(holiday_path, weekdays=(0, 1, 2, 3, 4), *, name=None):
    holiday_cal = load_calendar_file(holiday_path)
    weekday_cal = make_weekday_calendar(holiday_cal.range, weekdays)
    cal = weekday_cal & ~holiday_cal
    cal.name = (
        ",".join( str(w) for w in weekdays ) + " except " + holiday_cal.name
        if name is None
        else name
    )
    return cal


def format_calendar(cal) -> Iterable[str]:
    """
    Formats `cal` in the calendar file format.
    """
    start, stop = cal.range
    yield f"START {start}"
    yield f"STOP  {stop}"
    yield ""
    for date in cal.dates_array:
        yield str(date)


def dump_calendar_file(cal, path):
    """
    Writes `cal` as a calendar file at `path`.
    """
    with open(path, "wt") as file:
        for line in format_calendar(cal):
            print(line, file=file)


#-------------------------------------------------------------------------------

class CalendarDir:
    """
    A directory containing calendar files.

    Each calendar file has the suffix '.cal'.
    """

    SUFFIX = ".cal"

    def __init__(self, path):
        self.__path = Path(path)


    @property
    def path(self):
        """
        The path to the calendar directory.
        """
        return self.__path


    # FIXME: Caching?

    def __getitem__(self, name):
        """
        Gets a calendar from a calendar file.
        """
        path = (self.__path / name).with_suffix(self.SUFFIX)
        try:
            return load_calendar_file(path)
        except FileNotFoundError:
            raise KeyError(name)


# The initial calendar dir is the one shipped with Ora, or pointed to by
# ORA_CALENDARS if this is set.
try:
    _CALENDAR_DIR = os.environ["ORA_CALENDARS"]
except KeyError:
    _CALENDAR_DIR = Path(__file__).parent / "calendars"
_CALENDAR_DIR = CalendarDir(_CALENDAR_DIR)

def get_calendar_dir():
    """
    Returns the global calendar directory.
    """
    return _CALENDAR_DIR.path


def set_calendar_dir(path):
    """
    Sets the global calendar directory.
    """
    global _CALENDAR_DIR
    _CALENDAR_DIR = CalendarDir(path)


def _get_special_calendar(name):
    if name == "all":
        return make_const_calendar((Date.MIN, Date.MAX), True)
    if name == "none":
        return make_const_calendar((Date.MIN, Date.MAX), False)

    try:
        weekdays = parse_weekdays(name)
    except ValueError:
        pass
    else:
        cal = make_weekday_calendar((Date.MIN, Date.MAX), weekdays)
        # FIXME: Do this in make_weekday_calendar.
        cal.name = name
        return cal

    raise LookupError(f"unknown calendar: {name}")


def get_calendar(name):
    """
    Gets a calendar.

    The name may be:
    - "all" or "none"
    - A weekday expression; see `parse_weekdays`.
    - The name of a calendar in the global calendar directory.
    """
    name = str(name)
    try:
        return _get_special_calendar(name)
    except LookupError:
        pass

    return _CALENDAR_DIR[name]


