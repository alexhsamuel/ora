import pytest

from   ora import parse_daytime, Daytime

#-------------------------------------------------------------------------------

def test_hms():
    assert parse_daytime("%H:%M:%S", "23:06:09") == Daytime(23,  6,  9)
    assert parse_daytime("%H:%M:%S", "3:06:09" ) == Daytime( 3,  6,  9)
    assert parse_daytime("%H:%M:%S", "03:06:09") == Daytime( 3,  6,  9)
    assert parse_daytime("%H-%M=%S", "23-06=09") == Daytime(23,  6,  9)
    assert parse_daytime("%H%M%S"  , "230609"  ) == Daytime(23,  6,  9)
    assert parse_daytime("%M:%S:%H", "06:09:23") == Daytime(23,  6,  9)
    assert parse_daytime("it is %H:%M:%S now", "it is 23:06:09 now") == Daytime(23,  6,  9)

    assert parse_daytime("%H:%M:%S", "00:00:00") == Daytime( 0,  0,  0)
    assert parse_daytime("%H:%M:%S", "23:59:59") == Daytime(23, 59, 59)

    # Literal '%' in pattern.
    assert parse_daytime("%H%M%S%%", "230609%" ) == Daytime(23,  6,  9)


def test_hm():
    assert parse_daytime("%H:%M", "23:06") == Daytime(23,  6,  0)
    

def test_hms_fractional():
    assert parse_daytime("%H:%M:%S", "23:06:09."   ) == Daytime(23,  6,  9.0)
    assert parse_daytime("%H:%M:%S", "23:06:09.0"  ) == Daytime(23,  6,  9.0)
    assert parse_daytime("%H:%M:%S", "23:06:09.0000000") == Daytime(23,  6,  9.0)
    assert parse_daytime("%H:%M:%S", "23:06:09.125") == Daytime(23,  6,  9.125)


def test_hms_invalid():
    with pytest.raises(ValueError):
        parse_daytime("%H:M:%S", "24:00:00")
    with pytest.raises(ValueError):
        parse_daytime("%H:%M:%S", "102:00:00")
    with pytest.raises(ValueError):
        parse_daytime("%H:%M:%S", "23:60:05")
    with pytest.raises(ValueError):
        parse_daytime("%H:%M:%S", "23:00:60")
    with pytest.raises(ValueError):
        parse_daytime("%H:%M:%S", "99:99:99")
    with pytest.raises(ValueError):
        parse_daytime("%H:%M:%S", "233009")
    with pytest.raises(ValueError):
        parse_daytime("%H%M%S", "23:30:09")
    with pytest.raises(ValueError):
        parse_daytime("%H:%M", "23:30:09")
    with pytest.raises(ValueError):
        parse_daytime("%H:%M:%S", "23:30")

    # Missing digits.
    with pytest.raises(ValueError):
        parse_daytime("%H:%M:%S", "23:6:9")
    with pytest.raises(ValueError):
        parse_daytime("%H%M%S", "2369")


def test_usec():
    # FIXME: %S consumes the frational part if we use '.'.
    assert parse_daytime("%H:%M:%S/%f", "23:06:09/123456") == Daytime(23, 6, 9.123456)
    assert parse_daytime("%H:%M:%S/%f", "23:06:09/1") == Daytime(23, 6, 9.1)
    assert parse_daytime("%H:%M:%S/%f", "23:06:09/123") == Daytime(23, 6, 9.123)


def test_12hour():
    assert parse_daytime("%I:%M", "1:00"        ) == Daytime( 1,  0,  0)
    assert parse_daytime("%I:%M", "11:00"       ) == Daytime(11,  0,  0)
    assert parse_daytime("%I:%M:%S", "11:23:45" ) == Daytime(11,  23, 45)

    assert parse_daytime("%I:%M %p", "1:00 am"  ) == Daytime( 1,  0,  0)
    assert parse_daytime("%I:%M %p", "11:00 am" ) == Daytime(11,  0,  0)
    assert parse_daytime("%I:%M %p", "1:00 AM"  ) == Daytime( 1,  0,  0)
    assert parse_daytime("%I:%M %p", "11:00 AM" ) == Daytime(11,  0,  0)
    assert parse_daytime("it's %I:%M %p now", "it's 11:00 AM now") == Daytime(11,  0,  0)

    assert parse_daytime("%I:%M %p", "1:00 pm"  ) == Daytime(13,  0,  0)
    assert parse_daytime("%I:%M %p", "11:00 pm" ) == Daytime(23,  0,  0)
    assert parse_daytime("%I:%M %p", "1:00 PM"  ) == Daytime(13,  0,  0)
    assert parse_daytime("%I:%M %p", "11:00 PM" ) == Daytime(23,  0,  0)

    assert parse_daytime("%I:%M %p", "12:00 AM" ) == Daytime( 0,  0,  0)
    assert parse_daytime("%I:%M %p", "12:15 AM" ) == Daytime( 0, 15,  0)
    assert parse_daytime("%I:%M %p",  "1:15 AM" ) == Daytime( 1, 15,  0)
    assert parse_daytime("%I:%M %p", "11:15 AM" ) == Daytime(11, 15,  0)
    assert parse_daytime("%I:%M %p", "12:15 PM" ) == Daytime(12, 15,  0)
    assert parse_daytime("%I:%M %p",  "1:15 PM" ) == Daytime(13, 15,  0)
    assert parse_daytime("%I:%M %p", "11:15 PM" ) == Daytime(23, 15,  0)
    assert parse_daytime("%I:%M %p", "11:59 PM" ) == Daytime(23, 59,  0)


@pytest.mark.parametrize("fmt", (
    "%H%M%.3S",
    "%H:%M:%.3S",
    "%.3C",
    "%~.3C",
    "%.4~C",
    "%.5C",
    "%I:%M:%.3S %p",
    "%I:%M:%.3S %_p",
    "%I:%M:%.3S %^p",
))
def test_parse_roundtrip(fmt):
    y = Daytime(12, 30, 45.125)
    str = format(y, fmt)
    print(f"{fmt} → {str}")
    assert parse_daytime(fmt, str) == y


