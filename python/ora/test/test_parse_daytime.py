import pytest

from   ora import parse_daytime, Daytime

#-------------------------------------------------------------------------------

def test_hms():
    assert parse_daytime("%H:%M:%S", "23:06:09") == Daytime(23,  6,  9)
    assert parse_daytime("%H%M%S"  , "230609"  ) == Daytime(23,  6,  9)
    assert parse_daytime("%M:%S:%H", "06:09:23") == Daytime(23,  6,  9)


def test_hm():
    assert parse_daytime("%H:%M", "23:06") == Daytime(23,  6,  0)
    

@pytest.mark.xfail
def test_hms_fractional():
    assert parse_daytime("%H:%M:%S", "23:06:09."   ) == Daytime(23,  6,  9.0)
    assert parse_daytime("%H:%M:%S", "23:06:09.0"  ) == Daytime(23,  6,  9.0)
    assert parse_daytime("%H:%M:%S", "23:06:09.0000000") == Daytime(23,  6,  9.0)
    assert parse_daytime("%H:%M:%S", "23:06:09.125") == Daytime(23,  6,  9.125)


def test_hms_invalid():
    with pytest.raises(ValueError):
        parse_daytime("%H:%M:%S", "23:6:9")
    with pytest.raises(ValueError):
        parse_daytime("%H%M%S", "2369")


