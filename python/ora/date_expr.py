import re
import sys

from   . import Date, TimeZone, today

#-------------------------------------------------------------------------------

# Date calculation mini-language

# DATE [ +|- OFFSET ] [ # CAL ] [ @ TZ ]
#
# DATE: LITERAL | 'today' | 'next' | 'last'

REGEX = re.compile(
    r"""
    (
      [^-+#@]*? 
    )
    \s*
    (?:
      ( [-+] ) \s* ( \d+? )
    ) ?
    \s*
    (?: [#] \s* 
      ( [^#@]+? ) 
    ) ?
    \s*
    (?: @ \s* 
      ( [^@]+? ) 
    ) ?
    $
    """,
    re.VERBOSE)


class ParseError(Exception):
    """
    Failed to parse the date expression.
    """
    pass



def parse_expr(expr, Date=Date):
    expr = str(expr)
    match = REGEX.match(expr)
    if match is None:
        raise ParseError("invalid expression: {!r}".format(expr))

    date, sign, offset, calendar, time_zone = match.groups()
    offset = int(offset)

    if calendar is not None:
        # FIXME
        pass

    if time_zone is None:
        # FIXME
        pass
    elif time_zone == "local":
        # FIXME
        pass
    else:
        time_zone = TimeZone(time_zone)

    if date == "today":
        date = today(time_zone, Date=Date)
    elif date == "next":
        # FIXME
        pass
    elif date == "last":
        # FIXME
        pass
    else:
        date = Date(date)

    if offset is not None:
        # FIXME: Calendar.
        date += (1 if sign == "+" else -1) * int(offset)

    return date
        

if __name__ == "__main__":
    print(parse_expr(sys.argv[1]))


