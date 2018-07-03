from   pathlib import Path
from   .ext import make_weekday_calendar, parse_calendar

#-------------------------------------------------------------------------------

def load_calendar_file(path, *, name=None):
    """
    Loads a calendar from the file at `path`.

    The file has the following format::

        START date
        END date
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


