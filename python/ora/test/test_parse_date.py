import pytest

from   ora import parse_date, Date

#-------------------------------------------------------------------------------

def test_ymd():
    assert parse_date("%Y-%m-%d", "2018-02-11") == Date(2018,  2, 11)
    assert parse_date("%Y-%m-%d", "2018-2-11" ) == Date(2018,  2, 11)

    assert parse_date("%Y%m%d", "20180211"    ) == Date(2018,  2, 11)
    assert parse_date("%m/%d/%Y", "2/28/2018" ) == Date(2018,  2, 28)
    assert parse_date("%Y%m%d", "20180228"    ) == Date(2018,  2, 28)

    assert parse_date("%Y-%m-%d", "1-1-1"     ) == Date(   1,  1,  1)
    assert parse_date("%Y-%m-%d", "9999-12-31") == Date(9999, 12, 31)


def test_iso():
    assert parse_date("%D", "2018-02-11") == Date(2018,  2, 11)
    assert parse_date("%D", "2018-2-11" ) == Date(2018,  2, 11)

    assert parse_date("%D", "1-1-1"     ) == Date(   1,  1,  1)
    assert parse_date("%D", "9999-12-31") == Date(9999, 12, 31)


def test_ymd_2digit():
    assert parse_date("%m/%d/%y", "2/28/18" ) == Date(2018,  2, 28)
    assert parse_date("%y%m%d", "180228"    ) == Date(2018,  2, 28)
    assert parse_date("%y-%m-%d", "68-12-31") == Date(2068, 12, 31)
    assert parse_date("%y-%m-%d", "69-01-01") == Date(1969,  1,  1)


def test_month_name():
    assert parse_date("%Y %B %d", "2018 February 11") == Date(2018,  2, 11)
    assert parse_date("%Yx%Bx%d", "2018xDecemberx31") == Date(2018, 12, 31)

    with pytest.raises(ValueError):
        parse_date("%Y %B %d", "2018 Wednesday 11")
    with pytest.raises(ValueError):
        parse_date("%Y %B %d", "2018 Feb 11")
    with pytest.raises(ValueError):
        parse_date("%Y %B %d", "2018 Febuary 11")
    with pytest.raises(ValueError):
        parse_date("%Y %B %d", "2018 Februaryy 11")
    with pytest.raises(ValueError):
        parse_date("%Y %B %d", "2018 11")


def test_month_abbr():
    assert parse_date("%Y %b %d", "2018 Jan 1")  == Date(2018,  1,  1)
    assert parse_date("%Yx%bx%d", "2018xOctx31") == Date(2018, 10, 31)

    with pytest.raises(ValueError):
        parse_date("%Y %b %d", "2018 Tue 11")
    with pytest.raises(ValueError):
        parse_date("%Y %b %d", "2018 February 11")
    with pytest.raises(ValueError):
        parse_date("%Y %b %d", "2018 FEB 11")
    with pytest.raises(ValueError):
        parse_date("%Y %b %d", "2018 Fed 11")
    with pytest.raises(ValueError):
        parse_date("%Y %b %d", "2018 11")


def test_ymd_mismatch():
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "20180211")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-02:11")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-02-11x")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "20-18-02-11")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-x02-11")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018--11")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-Jan-11")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2/11/2018")


def test_ymd_range():
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-02-0")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-02-00")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-02-29")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-03-99")

    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-0-11")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-00-11")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-13-11")
    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "2018-100-11")

    with pytest.raises(ValueError):
        parse_date("%Y-%m-%d", "10000-02-11")


def test_week_date():
    assert parse_date("%G-%V-%u", "2018-06-7") == Date(2018,  2, 11)
    assert parse_date("%G-%V-%u", "2018-52-7") == Date(2018, 12, 30)
    assert parse_date("%G-%V-%u", "2019-01-1") == Date(2018, 12, 31)
    assert parse_date("%G-%V-%u", "2019-01-2") == Date(2019,  1,  1)


def test_weekday_name():
    assert parse_date("%G:%V/%A", "2018:8/Monday"   ) == Date(2018,  2, 19)
    assert parse_date("%G:%V/%A", "2018:8/Wednesday") == Date(2018,  2, 21)
    assert parse_date("%G:%V/%A", "2018:8/Sunday"   ) == Date(2018,  2, 25)
    assert parse_date("%G:%V/%A", "2018:9/Monday"   ) == Date(2018,  2, 26)


def test_weekday_abbr():
    assert parse_date("%a %V %G", "Mon 8 2018") == Date(2018,  2, 19)
    assert parse_date("%a %V %G", "Wed 8 2018") == Date(2018,  2, 21)
    assert parse_date("%a %V %G", "Sun 8 2018") == Date(2018,  2, 25)
    assert parse_date("%a %V %G", "Mon 9 2018") == Date(2018,  2, 26)


def test_ordinal():
    assert parse_date("%Y-%j", "2018-001") == Date(2018,  1,  1)
    assert parse_date("%Y**%j", "2018**1") == Date(2018,  1,  1)
    assert parse_date("%Y%j" , "2018365" ) == Date(2018, 12, 31)
    assert parse_date("%Y-%j", "2020-366") == Date(2020, 12, 31)


@pytest.mark.parametrize("fmt", (
    "%Y%m%d",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%B/%d/%Y",
    "%b/%d/%Y",
    "%~B/%d/%Y",
    "%D",
    "%~D",
    "%A %D",
    "%~A %D",
    "%^A %D",
    "%^a %D",
    "%_A %D",
    "%~_A %D",
    "%a %D",
    "%G-W%V-%u",
    "%Y-%j",
))
def test_parse_roundtrip(fmt):
    d = Date(2020, 1, 28)
    str = format(d, fmt)
    print(f"{fmt} → {str}")
    assert parse_date(fmt, str) == d


