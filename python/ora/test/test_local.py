from   ora import Date, Daytime, UTC

#-------------------------------------------------------------------------------

def test_date_daytime():
    d = Date(2020, 1, 16)
    y = Daytime(4, 37, 13.25)
    t = (d, y) @ UTC

    l = t @ UTC
    assert l.date == d
    assert l.daytime == y


def test_attrs():
    d = Date(2020, 1, 16)
    y = Daytime(4, 37, 13.25)
    t = (d, y) @ UTC

    l = t @ UTC

    assert l.year == d.year
    assert l.month == d.month
    assert l.day == d.day
    assert l.datenum == d.datenum
    assert l.ordinal == d.ordinal
    assert l.week == d.week
    assert l.week_date == d.week_date
    assert l.week_year == d.week_year
    assert l.weekday == d.weekday
    assert l.ymdi == d.ymdi

    assert l.hour == y.hour
    assert l.minute == y.minute
    assert l.second == y.second
    assert l.daytick == y.daytick
    assert l.ssm == y.ssm


def test_floor_div_date():
    d = Date(2020, 1, 16)
    y = Daytime(4, 37, 13.25)

    l = d // y
    assert tuple(l) == (d, y)

    l = d // "04:37:13.25"
    assert tuple(l) == (d, y)

    l == d // (4, 37, 13.25)
    assert tuple(l) == (d, y)


def test_floor_div_daytime():
    d = Date(2020, 1, 16)
    y = Daytime(4, 37, 13.25)

    l = d // y
    assert tuple(l) == (d, y)

    l = 20200116 // y
    assert tuple(l) == (d, y)

    l == "2020-01-16" // y
    assert tuple(l) == (d, y)

    l == (2020, 1, 16) // y
    assert tuple(l) == (d, y)


