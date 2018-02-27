def assert_float_equal(val0, val1):
    assert abs(val0 - val1) < max(abs(val0), abs(val1)) * 1e-12


def xeq(x0, x1, places=15):
    """
    Returns true iff `x0` and `x1` are within decimal `places` of each other.
    """
    scale = max(abs(x0), abs(x1))
    epsilon = scale * 10 ** -places
    return abs(x0 - x1) < epsilon


