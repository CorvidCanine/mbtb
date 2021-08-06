import pytest

import mbtb
import hdf5storage
import numpy as np


def test_gaussian():
    test_data = hdf5storage.read(filename="tests/test_data/gaussian_start.h5")
    test_grid = np.linspace(0, 100, 101)

    # Does the output match past output?
    assert (
        mbtb.gaussian(test_grid, 10, 50, 5, 0) == test_data["gaussian_start_zero_base"]
    ).all()

    assert (
        mbtb.gaussian(test_grid, 10, 50, 5, 10) == test_data["gaussian_start_10_base"]
    ).all()

    # The maximum value should always be the height of the gaussian
    assert mbtb.gaussian(test_grid, 100, 50, 10, 0).max() == 100
    # Maximum value should be the height of the gaussian even when off-centre and wide
    assert mbtb.gaussian(test_grid, 500, 20, 100, 0).max() == 500
    # The base should lift the gaussian height by that amount
    assert mbtb.gaussian(test_grid, 500, 20, 100, 100).max() == 600

    # At later times the maximum should be less than the starting height
    assert mbtb.gaussian(test_grid, 500, 20, 100, 0, time=10).max() < 500
