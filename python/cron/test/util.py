def assert_float_equal(val0, val1):
    assert abs(val0 - val1) < max(abs(val0), abs(val1)) * 1e-12


