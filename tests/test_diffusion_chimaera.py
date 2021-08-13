import pytest

import mbtb
import hdf5storage
import numpy as np


class DummeyChimaeraGrid:
    def __init__(self):
        self.overlaps = []
        self.grids = []

def test_diffusion_chimaera_with_zeros():
    base_grid = mbtb.Grid(
        "base",
        0,
        10,
        0.01,
        alpha=0.005,
        left_boundary=mbtb.Boundary.CONSTANT,
        right_boundary=mbtb.Boundary.CONSTANT,
    )
    base_grid.set_constant_boundary_values(left=0, right=0)
    base_grid.set_grid_start(0)
    dummey_chimaera = DummeyChimaeraGrid()
    dummey_chimaera.grids.append(base_grid)

    step = mbtb.diffusion_chimaera(0, np.zeros(1000), dummey_chimaera)

    # Running diffusion_chimaera with a y of just zeros should output just zeros
    assert step.sum() == 0

    base_grid = mbtb.Grid(
        "base",
        0,
        1,
        0.01,
        alpha=0.005,
        left_boundary=mbtb.Boundary.PERIODIC,
        right_boundary=mbtb.Boundary.PERIODIC,
    )
    base_grid.set_grid_start(0)
    dummey_chimaera = DummeyChimaeraGrid()
    dummey_chimaera.grids.append(base_grid)

    step = mbtb.diffusion_chimaera(0, np.zeros(100), dummey_chimaera)

    # Running diffusion_chimaera with a y of just zeros should output just zeros
    # also applies to periodic boundaries, with a smaller domain
    assert step.sum() == 0

    overlap_grid = mbtb.Grid("overlap", 0.5, 0.6, 0.005, alpha=0.005)
    overlap_grid.set_grid_start(100)
    dummey_chimaera.grids.append(overlap_grid)
    step = mbtb.diffusion_chimaera(0, np.zeros(120), dummey_chimaera)

    # Adding an overlapped grid on to the base grid should also still only
    # output zeros
    assert step.sum() == 0
