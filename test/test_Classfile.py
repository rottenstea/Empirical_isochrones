from EmpiricalArchive.Extraction.Classfile import *
import numpy as np


def test_abs_mag_error():
    # Test with known values
    w = 10.0
    delta_w = 0.1
    delta_m = 0.05
    expected_result = np.sqrt((5 / (np.log(10) * w) * delta_w) ** 2 + delta_m ** 2)
    assert np.isclose(abs_mag_error(w, delta_w, delta_m), expected_result)


def test_RSS():
    # Test with known values
    e1 = 0.1
    e2 = 0.2
    expected_result = np.sqrt(e1 ** 2 + e2 ** 2)
    assert np.isclose(RSS(e1, e2), expected_result)