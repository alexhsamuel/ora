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


