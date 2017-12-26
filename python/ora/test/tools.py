def xeq(x0, x1, places=15):
    """
    Returns true iff `x0` and `x1` are within decimal `places` of each other.
    """
    scale = max(abs(x0), abs(x1))
    epsilon = scale * 10 ** -places
    return abs(x0 - x1) < epsilon


