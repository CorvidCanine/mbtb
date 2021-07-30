import pytest

import mbtb
import hdf5storage
import numpy as np


def test_gaussian_start():
    test_data = hdf5storage.read(filename="tests/test_data/gaussian_start.h5")
    test_grid = np.linspace(0, 100, 101)

    assert (
        mbtb.gaussian_start(test_grid, 10, 50, 5, 0)
        == test_data["gaussian_start_zero_base"]
    ).all()

    assert (
        mbtb.gaussian_start(test_grid, 10, 50, 5, 10)
        == test_data["gaussian_start_10_base"]
    ).all()
