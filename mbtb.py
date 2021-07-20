import time
import argparse
import warnings
import toml
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib import style
from matplotlib import animation
from collections import namedtuple
from enum import Enum, auto
from functools import partial
from pathlib import Path

try:
    # Load custom matplotlib style if it's avalible
    style.use(["nord-base-small", "corvid-light"])
except OSError:
    pass


class Boundary(Enum):
    PERIODIC = auto()
    WALL = auto()
    # IGNORE = auto()
    CONSTANT = auto()


class Overlap:
    """Class for handling the overlap of grids.

    Attributes`
    ----------
    lower_grid : Grid
        The lower grid
    over_grid : Grid
        The grid that is on top of the lower grid
    num_fringe_cells : int
        The number of cells to be interpolated from the other grid
    interface_width : float
        The width of the interface at the ends of the overlap
    [lower/over]_[left/right]_interp_range : tuple of ints, shape (2)
        The range of cells to be used for interpolation to the other
        grid. Indexs are for within each grid, not the chimaera grid.
    [lower/over]_[left/right]_interp_positions : tuple of floats, shape (2)
        The range of positions of the cells to be used for interpolation
        to the other grid.
    [lower/over]_[left/right]_fringe_range : tuple of ints, shape (2)
        The range of cells to be interpolated too. Index are for
        within each grid.
    [lower/over]_[left/right]_fringe_positions : tuple of floats, shape (2)
        The range of positions of the cells to be interpolated too.
    """

    def __init__(self, lower_grid, over_grid, interface_width=0.03, num_fringe_cells=1):
        """Initialisation method for the Overlap class.

        Parameters
        ----------
        lower_grid : Grid
            The lower grid.
        over_grid : Grid
            The grid of top of the lower grid.
        interface_width : float, optional
            The width of the interface at the ends of the overlap
            region, by default 0.03
        num_fringe_cells : int, optional
            The number of cells to be interpolated to from the
            other grid, by default 1

        Raises
        ------
        ValueError
            If num_fringe_cells is less than 1
        """

        self.lower_grid = lower_grid
        self.over_grid = over_grid

        if num_fringe_cells < 1:
            raise ValueError("The number of overlap cells must be at least 1")

        self.num_fringe_cells = num_fringe_cells
        self.interface_width = interface_width

        left_cut_in_lower_index, right_cut_in_lower_index = self.lower_grid.cut_hole(
            self.over_grid.left_pos, self.over_grid.right_pos, self.interface_width
        )

        # Lower grid, left side
        left_pos_in_lower_index = self.lower_grid.position_to_cell(
            self.over_grid.left_pos
        )
        self.lower_left_interp_range = (
            left_pos_in_lower_index - 1,
            left_cut_in_lower_index - num_fringe_cells,
        )
        self.lower_left_interp_positions = self.lower_grid.cell_positions[
            self.lower_left_interp_range[0] : self.lower_left_interp_range[1]
        ]
        self.lower_left_fringe_range = (
            left_cut_in_lower_index - num_fringe_cells,
            left_cut_in_lower_index,
        )
        self.lower_left_fringe_positions = self.lower_grid.cell_positions[
            self.lower_left_fringe_range[0] : self.lower_left_fringe_range[1]
        ]

        # Overlaped grid, left side
        left_cut_in_over_index = self.over_grid.position_to_cell(
            self.over_grid.left_pos + self.interface_width
        )
        self.over_left_interp_range = (
            num_fringe_cells,
            left_cut_in_over_index + 1,
        )
        self.over_left_interp_positions = self.over_grid.cell_positions[
            self.over_left_interp_range[0] : self.over_left_interp_range[1]
        ]
        self.over_left_fringe_range = (0, num_fringe_cells)
        self.over_left_fringe_positions = self.over_grid.cell_positions[
            0:num_fringe_cells
        ]

        # Lower grid, right side
        right_pos_in_lower_index = self.lower_grid.position_to_cell(
            self.over_grid.right_pos
        )
        self.lower_right_interp_range = (
            right_cut_in_lower_index + num_fringe_cells,
            right_pos_in_lower_index + 2,
        )
        self.lower_right_interp_positions = self.lower_grid.cell_positions[
            self.lower_right_interp_range[0] : self.lower_right_interp_range[1]
        ]
        self.lower_right_fringe_range = (
            right_cut_in_lower_index,
            right_cut_in_lower_index + num_fringe_cells,
        )
        self.lower_right_fringe_positions = self.lower_grid.cell_positions[
            self.lower_right_fringe_range[0] : self.lower_right_fringe_range[1]
        ]

        # Overlaped grid, right side
        right_cut_in_over_index = self.over_grid.position_to_cell(
            self.over_grid.right_pos - self.interface_width
        )
        self.over_right_interp_range = (
            right_cut_in_over_index - 1,
            self.over_grid.num_cells - num_fringe_cells,
        )
        self.over_right_interp_positions = self.over_grid.cell_positions[
            self.over_right_interp_range[0] : self.over_right_interp_range[1]
        ]
        self.over_right_fringe_range = (
            self.over_grid.num_cells - num_fringe_cells,
            self.over_grid.num_cells,
        )
        self.over_right_fringe_positions = self.over_grid.cell_positions[
            self.over_right_fringe_range[0] : self.over_right_fringe_range[1]
        ]

    def __str__(self):
        return f"Overlap of grid {self.over_grid.index} on lower grid {self.lower_grid.index} with {self.num_fringe_cells} fringe cells"


class ChimaeraGrid:
    """
    A class for working with chimaera/multiblock grids.

    This class holds the multiple 1d grids (stored as Grid objects)
    that make up a chimaera grid and the Overlap objects that describe
    how those grids are interfaced to each other. It also calculates any
    needed infomation for the solver, and handles the running of the solver,
    which is scipy's solve_ivp function.

    Attributes
    ----------
    grids : list of Grid
        The grids that makeup this chimaera grid
    overlaps : list of Overlap
        The overlaps of the grids
    is_ready : bool
        Flag that is True after `ready` has been called
    is_solved : bool
        Flag that is `True` after `solve` has been called
    total_num_cells : int
        The number of cells in all of the grids,
        is only set after calling `ready`
    result : OdeResult
        The solution found by solve_ivp
    solver_elapsed_time: float
        The time the solver took, set after
        calling `solve`
    total_energy : float
        The total energy of all grids for each solved time step
    """

    def __init__(self):
        """Initialisation method for ChimaeraGrid

        Has no parameters.
        """

        self.grids = []
        self.overlaps = []
        self.starting_conditions = []
        self.cell_positions = np.array([])
        self.active_y = np.array([])
        self.total_num_cells = 0
        self.result = None
        self.solver_elapsed_time = None

        self.is_ready = False
        self.is_solved = False

    def add_grid(self, new_grid, interface_width=0.03, num_fringe_cells=1):
        """Registers a new grid with the ChimaeraGrid

        An Overlap will be created if any grids are below this new
        one. New grids are placed on top of previous grids.

        Parameters
        ----------
        new_grid : Grid
            The grid to be registered
        num_fringe_cells : int, optional
            The number of cells at the edge of overlaps to be
            interpolated to, by default 1
        """

        new_grid.index = len(self.grids)

        for lower_grid in self.grids:
            if lower_grid.does_pos_range_overlap(new_grid.left_pos, new_grid.right_pos):
                # If there is a grid underneath the new grid, create a Overlap
                self.overlaps.append(
                    Overlap(
                        lower_grid,
                        new_grid,
                        interface_width=interface_width,
                        num_fringe_cells=num_fringe_cells,
                    )
                )

        self.grids.append(new_grid)

    def ready(self, starting_condition=None):
        """Readies the ChimaeraGrid to be solved, should be called before `solve`.

        Calculates the positions of cells in each grid within
        the overall space and the total number of cells in the collection.
        Sets the starting conditions.

        The starting conditions dictionary should have a 'type' field that selectes
        the type of starting condition to run. This can be 'gaussian' or 'preset'.

        If 'gaussian', the parameters for the gaussian must be in the dict.
        If 'preset', 'starting_array' should be an array of floats of length num_cells
        that is a precalculated starting condition.

        Parameters
        ----------
        starting_condition : dict
            A dictionary specifying the starting conditions.
            If `None`, cell 40 will be set to 1000, this is the
            default.

        Raises
        ------
        Exception
            If no grids are registered when calling this function an
            exception is raised
        """

        if len(self.grids) == 0:
            raise Exception("At least one grid needs to be registerd before readying")

        counter = 0
        for grid in self.grids:
            grid.set_grid_start(counter)
            counter += grid.num_cells
            self.cell_positions = np.concatenate(
                (self.cell_positions, grid.cell_positions)
            )
            self.active_y = np.concatenate((self.active_y, grid.active))

        self.total_num_cells = counter

        # Convert from an array of 1s and 0s to bools
        self.active_y = self.active_y.astype("bool")
        starting_y = np.zeros(self.total_num_cells)

        if starting_condition is None:
            starting_y[40] = 1000
        elif starting_condition["type"] == "gaussian":
            starting_y[self.active_y] = gaussian_start(
                self.cell_positions[self.active_y],
                starting_condition["height"],
                starting_condition["centre"],
                starting_condition["width"],
                starting_condition["base"],
            )
        elif starting_condition["type"] == "preset":
            starting_y[self.active_y] = starting_condition["starting_array"]
        else:
            raise ValueError(
                f"Unknown starting condition type, {starting_condition.type}."
            )

        self.starting_y = starting_y
        self.is_ready = True

    def solve(self, solver_time_span, solver_method):
        """Solve the grid collection using scipy's solve_ivp.

        Will solve the grid collection for the specified time span.
        The solution will be stored in self.solver.
        The total energy is calculated and stored in self.total_energy.
        The `ready` method needs to be called before this function.

        Parameters
        ----------
        solver_time_span : 2-tuple of floats
            Interval of integration (t0, tf), for solve_ivp.
        solver_method : string or OdeSolver
            Integration method for solve_ivp to use.

        Raises
        ------
        Exception
            If the method `ready` hasn't been run an exception will be raised
        """

        if not self.is_ready:
            raise Exception("The ready method must be run before solving")

        # Store the arguments and solver method used
        self.solver_time = solver_time_span
        self.solver_method = solver_method

        solver_start_time = time.perf_counter()
        solver = solve_ivp(
            diffusion_chimaera,
            solver_time_span,
            self.starting_y,
            args=(self,),
            method=solver_method,
            # jac=J,
        )
        self.solver_elapsed_time = time.perf_counter() - solver_start_time

        self.total_energy = np.zeros(len(solver.t))
        for grid in self.grids:
            grid.give_solution(solver.y[grid.start : grid.end])
            self.total_energy += grid.energy

        print(f"Ran for {self.solver_elapsed_time:6.1f}s. {solver.message}")

        self.result = solver
        self.is_solved = True

    def print_energy_check(self):
        """Print the energy difference for each grid and the total to terminal"""

        if self.is_solved:
            print(f"{'Grid':^20}  Start energy   End energy   Energy difference")
            for grid in self.grids:
                print(
                    f"{grid.name:^20}  {grid.energy[0]:10.4f}     {grid.energy[-1]:10.4f}    {grid.energy[-1] - grid.energy[0]:10.4f}"
                )
            print(
                f"{'Total':^20}  {self.total_energy[0]:10.4f}     {self.total_energy[-1]:10.4f}    {self.total_energy[-1] - self.total_energy[0]:10.4f}"
            )
        else:
            print("This grid collection has not been solved yet")

    def scatter_plot(self, time_to_plot, overlap_markers=True):
        if not self.is_solved:
            warnings.warn("The solve method must be run before plotting")
            return

        fig, axs = plt.subplots(
            2, 1, sharex="col", gridspec_kw={"height_ratios": [3, 1]}
        )
        # The solver produces output at times it fixes is sutible,
        # so we need to find the closest time with output to the requested time
        time_to_plot_index = np.abs(self.result.t - time_to_plot).argmin()

        for grid in self.grids:
            active_bool = grid.active.astype(bool)
            axs[0].scatter(
                grid.cell_positions[active_bool],
                grid.solution[active_bool, time_to_plot_index],
                label=f"Grid: {grid.name}",
            )

            axs[1].barh(
                grid.name,
                grid.right_pos - grid.left_pos,
                left=grid.left_pos,
                height=0.25,
            )

        if overlap_markers:
            overlap_marker_colour = "#d3423e"
            fringe_marker = "s"
            interp_marker = "^"
            overlap_marker_size = 10
            for overlap in self.overlaps:
                axs[0].scatter(
                    overlap.lower_left_interp_positions,
                    overlap.lower_grid.solution[
                        overlap.lower_left_interp_range[
                            0
                        ] : overlap.lower_left_interp_range[1],
                        time_to_plot_index,
                    ],
                    marker=interp_marker,
                    s=overlap_marker_size,
                    c=overlap_marker_colour,
                )

                axs[0].scatter(
                    overlap.over_left_interp_positions,
                    overlap.over_grid.solution[
                        overlap.over_left_interp_range[
                            0
                        ] : overlap.over_left_interp_range[1],
                        time_to_plot_index,
                    ],
                    marker=interp_marker,
                    s=overlap_marker_size,
                    c=overlap_marker_colour,
                )

                axs[0].scatter(
                    overlap.lower_left_fringe_positions,
                    overlap.lower_grid.solution[
                        overlap.lower_left_fringe_range[
                            0
                        ] : overlap.lower_left_fringe_range[1],
                        time_to_plot_index,
                    ],
                    marker=fringe_marker,
                    s=overlap_marker_size,
                    c=overlap_marker_colour,
                )

                axs[0].scatter(
                    overlap.over_left_fringe_positions,
                    overlap.over_grid.solution[
                        overlap.over_left_fringe_range[
                            0
                        ] : overlap.over_left_fringe_range[1],
                        time_to_plot_index,
                    ],
                    marker=fringe_marker,
                    s=overlap_marker_size,
                    c=overlap_marker_colour,
                )

                axs[0].scatter(
                    overlap.lower_right_interp_positions,
                    overlap.lower_grid.solution[
                        overlap.lower_right_interp_range[
                            0
                        ] : overlap.lower_right_interp_range[1],
                        time_to_plot_index,
                    ],
                    marker=interp_marker,
                    s=overlap_marker_size,
                    c=overlap_marker_colour,
                )

                axs[0].scatter(
                    overlap.over_right_interp_positions,
                    overlap.over_grid.solution[
                        overlap.over_right_interp_range[
                            0
                        ] : overlap.over_right_interp_range[1],
                        time_to_plot_index,
                    ],
                    marker=interp_marker,
                    s=overlap_marker_size,
                    c=overlap_marker_colour,
                )

                axs[0].scatter(
                    overlap.over_right_fringe_positions,
                    overlap.over_grid.solution[
                        overlap.over_right_fringe_range[
                            0
                        ] : overlap.over_right_fringe_range[1],
                        time_to_plot_index,
                    ],
                    marker=fringe_marker,
                    s=overlap_marker_size,
                    c=overlap_marker_colour,
                )

                axs[0].scatter(
                    overlap.lower_right_fringe_positions,
                    overlap.lower_grid.solution[
                        overlap.lower_right_fringe_range[
                            0
                        ] : overlap.lower_right_fringe_range[1],
                        time_to_plot_index,
                    ],
                    marker=fringe_marker,
                    s=overlap_marker_size,
                    c=overlap_marker_colour,
                )

        axs[0].legend()
        axs[0].set_xlabel("Position [m]")
        axs[0].set_ylabel("Temperture [K]")
        axs[1].set_xlabel("Position [m]")
        axs[1].set_ylabel("Grid")
        axs[0].tick_params(reset=True)

        print(
            f"Showing scatter plot of each grid for time t={self.result.t[time_to_plot_index]:.3e}s"
        )

        plt.show()

    def __str__(self):
        return f"Grid Collection of {len(self.grids)} grids and {len(self.overlaps)} overlaps, has {'' if self.is_solved else 'not '}been solved"


class Grid:
    """A class for representing a 1d quadrilateral grid.

    Attributes
    ----------
    name : str
        The human readable name of this grid
    index : int
        The index of this grid within a ChimearaGrid.
        Is `None` if not a part of a larger grid.
    left_pos : float
        The left most position (Left edge of the first cell) of this grid
    right_pos: float
        The right most position (Right edge of the last cell) of this grid
    num_cells : int
        The total number of cells in this grid. Includes inactive and boundary
        cells.
    dx : ndarray, shape (num_cells)
        The cell width for every cell
    alpha : ndarray, shape (num_cells)
        The constant, alpha, for every cell
    cell_positions : ndarray, shape(num_cells)
        The position of the cell centre/node of every cell
    active : ndarray, shape(num_cells)
        An array specifying which cells will be updated by
        the solver (cells set to 1) and which will not be
        updated (cells set to 0)
    left_boundary : Boundary
        The type of the left boundary
    right_boundary : Boundary
        The type of the right boundary
    solution : ndarray, shape (num_cells, number of time points)
        2D array of the solved grid
        Set by `give_solution`.
    energy : ndarray, shape(number of time points)
        The total energy of the grid at each solved time point,
        calculated when `give_solution` is called
    jacobian : ndarray, shape(num_cells, num_cells)
        The jacobian for this grid
    left_boundary_value : Boundary
        The type of the left boundary
    right_boundary_value : Boundary
        The type of the right boundary
    start : int
        The left most index of this grid in the chimaera array
    end : int
        The right most index of this grid in the chimaera array
    """

    def __init__(
        self,
        name,
        left_pos,
        right_pos,
        dx,
        alpha=1,
        left_boundary=Boundary.WALL,
        right_boundary=Boundary.WALL,
    ):
        """Initialisation method for Grid.

        Calculates the number of cells in the grid from the arguments.
        Initialises the active array to all cells as active.
        Calculates the cell position of every cell within the domain.
        Calls the method to calculate the grids jacobian.

        Parameters
        ----------
        name : str
            The human readable name for this grid
        left_pos : float
            The left most position of this grid
            (Left edge of the first cell)
        right_pos : float
            The right most position of this grid
            (Right edge of the last cell)
        dx : float or array_like, shape (num_cells)
            Either a float for a uniform cell width or
            an array of the cell width for each cell. In the
            later case `num_cells` is set to the length of the
            passed array.
        alpha : float or array_like, shape (num_cells), optional
            Either a float for a uniform alpha constant or
            an array of the alpha value for each cell. In the
            later case the length of the passed array must match.
            The default is a uniform value of 1.
        left_boundary : Boundary, optional
            The type of the left boundary, by default Boundary.WALL
        right_boundary : Boundary, optional
            The type of the right boundary, by default Boundary.WALL

        Raises
        ------
        TypeError
            If a type other than a Boundary is passed to left_boundary
            or right_boundary a TypeError will be thrown.
        """

        self.name = name
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.index = None
        self.solution = None
        self.start = self.end = None
        self.energy = None

        # Check that left_boundary and right_boundary are from the Boundary enum
        # to avoid confusing errors later if they are not
        if isinstance(left_boundary, Boundary):
            self.left_boundary = left_boundary
        else:
            raise TypeError(
                f"left_boundary argument must be a Boundary type, not {type(left_boundary)}"
            )

        if isinstance(right_boundary, Boundary):
            self.right_boundary = right_boundary
        else:
            raise TypeError(
                f"right_boundary argument must be a Boundary type, not {type(right_boundary)}"
            )

        # I could use overloading instead of type checking, but this works for allowing
        # dx or alpha to both be either a float or an array
        if isinstance(dx, float):
            self.num_cells = int((self.right_pos - self.left_pos) / dx)
            self.dx = np.full(self.num_cells, dx, dtype=np.float64)
        else:
            self.num_cells = len(dx)
            self.dx = np.array(dx, dtype=np.float64)

        if isinstance(alpha, float) or isinstance(alpha, int):
            self.alpha = np.full(self.num_cells, alpha, dtype=np.float64)
        else:
            if len(alpha) != self.num_cells:
                raise ValueError(
                    "The length of the alpha array must match the number of cells in the grid."
                    + f" The passed array had length {len(alpha)}, the number of cells is {self.num_cells}."
                )
            self.alpha = np.array(alpha, dtype=np.float64)

        self.active = np.ones(self.num_cells)
        self.cell_positions = self.dx.cumsum() - self.dx / 2 + self.left_pos
        self._calc_jacobian()

    def __str__(self):
        return f"Grid {self.index}, from {self.left_pos} to {self.right_pos} with {self.num_cells} cells"

    def give_solution(self, solution):
        """Pass the solved grid to the grid object.

        Parameters
        ----------
        solution : ndarray, shape (num_cells, number of time points)
            Should be a 2d array of every cell in the grid at each solved time point.
            This is every cell in this grid object, not the entire domain.
        """

        self.solution = solution
        self.energy = np.sum(solution * self.dx[:, np.newaxis], axis=0)

    def set_constant_boundary_values(self, left=None, right=None):
        """If this grid has a constant boundary, the value of it can be specified

        A warning will be raised if a value is set for a non-constant boundary.

        Parameters
        ----------
        left : float, optional
            The value for the left hand boundary to be held at, by default None
        right : float, optional
            The value for the right hand boundary to be held at, by default None
        """

        if left:
            if self.left_boundary is Boundary.CONSTANT:
                self.left_boundary_value = left
            else:
                warnings.warn(
                    f"The left boundary is type {self.left_boundary}, so will not use the set value."
                )

        if right:
            if self.right_boundary is Boundary.CONSTANT:
                self.right_boundary_value = right
            else:
                warnings.warn(
                    f"The right boundary is type {self.right_boundary}, so will not use the set value."
                )

    def set_grid_start(self, start):
        """Set the index of the start of this grid within the overall y array.

        This tells the grid where in the full domain y array, that's passed to the
        solver, it begins. The end is assumed to be at plus the number of cells in
        the grid.

        Parameters
        ----------
        start : int
            The left most index of this grid in the domain array
        """

        self.start = start
        self.end = start + self.num_cells

    def position_to_cell(self, pos):
        """Converts a position in the domain to a cell index

        The index returned is the closest one to the given position
        and is for a cell within this grid object, not in the domain array.

        Parameters
        ----------
        pos : float
            The position to be converted

        Returns
        -------
        int
            The index of the cell closest to `pos`
        """

        return np.abs(self.cell_positions - pos).argmin()

    def left_right_cell_pos(self):
        """Gets the position of the left most and right most cell

        Returns
        -------
        2-tuple of floats
            Tuple of the left position and the right position
        """

        return self.cell_positions[0], self.cell_positions[-1]

    def cut_hole(self, left_cut_pos, right_cut_pos, interface_width):
        """Sets a range of the grid to inactive.

        Parameters
        ----------
        left_cut_pos : float
            The begining of the cut
        right_cut_pos : float
            The end of the cut
        interface_width : float
            The width of the region at the ends of the cut
            used for overlap interfacing.

        Returns
        -------
        2-tuple of ints
            The left and right index of the cut within this grid
        """

        left_cut_index = self.position_to_cell(left_cut_pos + interface_width)
        right_cut_index = self.position_to_cell(right_cut_pos - interface_width) + 1

        self.active[left_cut_index:right_cut_index] = 0

        return left_cut_index, right_cut_index

    def does_pos_range_overlap(self, leftmost_pos, rightmost_pos):
        """Checks if a position range overlaps with this grid.

        The range is checked against the defining range for this grid,
        which is the left edge of the first cell and the right edge of the last
        cell. The first cell centre/node will be half a cell width away.

        Parameters
        ----------
        leftmost_pos : float
            The left position of the range to check
        rightmost_pos : float
            The right position of the range to check

        Returns
        -------
        bool
            `True` if the range overlaps this grid,
            `False` if it does not overlap this grid.
        """

        return leftmost_pos <= self.right_pos and rightmost_pos >= self.left_pos

    def _calc_jacobian(self):
        """Internal method to calculate the grids jacobian matrix.

        This method is called by the grids initialisation method,
        it does not return the jacobian but stores it in `self.jacobian`.
        """

        jec = np.zeros((self.num_cells, self.num_cells))

        for i in range(1, self.num_cells - 1):

            left_alpha_grad = (self.alpha[i] + self.alpha[i - 1]) / 2
            jec[i, i - 1] = left_alpha_grad / ((self.dx[i - 1] + self.dx[i]) / 2) ** 2

            right_alpha_grad = (self.alpha[i] + self.alpha[i + 1]) / 2
            jec[i, i + 1] = right_alpha_grad / ((self.dx[i + 1] + self.dx[i]) / 2) ** 2

            jec[i, i] = -(jec[i, i - 1] + jec[i, i + 1])

        if self.left_boundary is Boundary.PERIODIC:
            left_alpha_grad = (self.alpha[0] + self.alpha[-1]) / 2
            jec[0, -1] = left_alpha_grad / ((self.dx[-1] + self.dx[0]) / 2) ** 2

            right_alpha_grad = (self.alpha[0] + self.alpha[1]) / 2
            jec[0, 1] = right_alpha_grad / ((self.dx[1] + self.dx[0]) / 2) ** 2

            jec[0, 0] = -(jec[0, -1] + jec[0, 1])
        elif self.left_boundary is Boundary.WALL:
            right_alpha_grad = (self.alpha[0] + self.alpha[1]) / 2
            jec[0, 1] = right_alpha_grad / ((self.dx[1] + self.dx[0]) / 2) ** 2
            jec[0, 0] = -jec[0, -1]

        if self.right_boundary is Boundary.PERIODIC:
            left_alpha_grad = (self.alpha[-1] + self.alpha[-2]) / 2
            jec[-1, -2] = left_alpha_grad / ((self.dx[-2] + self.dx[-1]) / 2) ** 2

            right_alpha_grad = (self.alpha[-1] + self.alpha[0]) / 2
            jec[-1, 0] = right_alpha_grad / ((self.dx[0] + self.dx[-1]) / 2) ** 2

            jec[-1, -1] = -(jec[-1, -2] + jec[-1, 0])
        elif self.right_boundary is Boundary.WALL:
            left_alpha_grad = (self.alpha[-1] + self.alpha[-2]) / 2
            jec[-1, -2] = left_alpha_grad / ((self.dx[-2] + self.dx[-1]) / 2) ** 2
            jec[-1, 0] = -jec[-1, -2]

        self.jacobian = jec


def gaussian_start(positions, height, centre, width, base):
    """Creates a guassian starting condition.

    Parameters
    ----------
    positions : ndarray, shape(num_cells)
        The position of the cell centre/node of every cell
    height : float
        a, The maximum value of the gaussian.
        This is in kelvin if solving the diffusion equ.
    centre : float
        b or x0, the position the gaussian will
        be centred on.
    width : float
        c or σ, is related to how wide the
        gaussian is.
    base : float
        A base constant that the gaussian sits upon

    Returns
    -------
    ndarray, shape(num_cells)
        A starting array of values
    """

    return base + height * np.exp(-((positions - centre) ** 2) / (2 * width * width))


def diffusion_fixed(t, y, alpha, dx):

    y1 = np.zeros(len(y))

    y[0] = 100
    y[-1] = 0

    for i in range(1, len(y) - 1):
        y1[i] = alpha * (y[i - 1] - 2 * y[i] + y[i + 1]) / dx ** 2

    y1[0] = 100
    y1[-1] = 0

    return y1


def diffusion_periodic(t, y, alpha, dx):

    y1 = np.zeros(len(y))

    for i in range(1, len(y) - 1):
        # y1[i] = alpha * (y[i - 1] - 2 * y[i] + y[i + 1]) / ((dx[i-1]+dx[i+1])/2 + dx[i]) ** 2
        y1[i] = (alpha[i] * (y[i + 1] - y[i]) / ((dx[i + 1] + dx[i]) / 2) ** 2) + (
            alpha[i] * (y[i - 1] - y[i]) / ((dx[i - 1] + dx[i]) / 2) ** 2
        )
        # print(f"cell {i}, left {(dx[i-1] + dx[i]) / 2}, right {(dx[i+1] + dx[i]) / 2 }")

    y1[0] = (alpha[0] * (y[1] - y[0]) / ((dx[1] + dx[0]) / 2) ** 2) + (
        alpha[0] * (y[-1] - y[0]) / ((dx[-1] + dx[0]) / 2) ** 2
    )
    y1[-1] = (alpha[-1] * (y[0] - y[-1]) / ((dx[0] + dx[-1]) / 2) ** 2) + (
        alpha[-1] * (y[-2] - y[-1]) / ((dx[-2] + dx[-1]) / 2) ** 2
    )

    return y1


def diffusion_chimaera(t, y, chimaera_grid):
    """
    Calculates a single step of diffusion on a 1d chimaera grid.

    Parameters
    ----------
    t : float
        The time of this step, unused by required by solve_ivp
    y : ndarray, shape(num_cells)
        Array of all cells in the chimaera grid
    chimaera_grid : ChimaeraGrid
        The description of the grid

    Returns
    -------
    ndarray, shape(num_cells)
        The y grid differentiated by one step
    """

    y1 = np.zeros(len(y))

    for ovlp in chimaera_grid.overlaps:

        # Lower grid interpolate to overlap grid, left side
        interp_func = interp1d(
            ovlp.lower_left_interp_positions,
            y[
                ovlp.lower_grid.start
                + ovlp.lower_left_interp_range[0] : ovlp.lower_grid.start
                + ovlp.lower_left_interp_range[1]
            ],
        )
        y[
            ovlp.over_grid.start
            + ovlp.over_left_fringe_range[0] : ovlp.over_grid.start
            + ovlp.over_left_fringe_range[1]
        ] = interp_func(ovlp.over_left_fringe_positions)

        # Overlaped grid interp to lower grid, left side
        interp_func = interp1d(
            ovlp.over_left_interp_positions,
            y[
                ovlp.over_grid.start
                + ovlp.over_left_interp_range[0] : ovlp.over_grid.start
                + ovlp.over_left_interp_range[1]
            ],
        )
        y[
            ovlp.lower_grid.start
            + ovlp.lower_left_fringe_range[0] : ovlp.lower_grid.start
            + ovlp.lower_left_fringe_range[1]
        ] = interp_func(ovlp.lower_left_fringe_positions)

        # Lower grid interp to overlap grid, right side
        interp_func = interp1d(
            ovlp.lower_right_interp_positions,
            y[
                ovlp.lower_grid.start
                + ovlp.lower_right_interp_range[0] : ovlp.lower_grid.start
                + ovlp.lower_right_interp_range[1]
            ],
        )
        y[
            ovlp.over_grid.start
            + ovlp.over_right_fringe_range[0] : ovlp.over_grid.start
            + ovlp.over_right_fringe_range[1]
        ] = interp_func(ovlp.over_right_fringe_positions)

        # Overlaped grid interp to lower grid, right side
        interp_func = interp1d(
            ovlp.over_right_interp_positions,
            y[
                ovlp.over_grid.start
                + ovlp.over_right_interp_range[0] : ovlp.over_grid.start
                + ovlp.over_right_interp_range[1]
            ],
        )
        y[
            ovlp.lower_grid.start
            + ovlp.lower_right_fringe_range[0] : ovlp.lower_grid.start
            + ovlp.lower_right_fringe_range[1]
        ] = interp_func(ovlp.lower_right_fringe_positions)

    for grid in chimaera_grid.grids:

        left_alpha = (grid.alpha[1:] + grid.alpha[:-1]) / 2
        left_J = (
            np.diagonal(grid.jacobian)[1:] + np.diagonal(grid.jacobian, offset=-1)
        ) / 2
        left_gradient = (
            2
            * (y[grid.start : grid.end - 1] - y[grid.start + 1 : grid.end])
            / (grid.dx[:-1] + grid.dx[1:])
        )
        left_flux = left_alpha * left_J * left_gradient

        y1[grid.start + 1 : grid.end] += (
            left_flux / (grid.dx[1:] * left_J) * grid.active[1:] * grid.active[:-1]
        )

        right_alpha = (grid.alpha[:-1] + grid.alpha[1:]) / 2
        right_J = (
            np.diagonal(grid.jacobian)[:-1] + np.diagonal(grid.jacobian, offset=1)
        ) / 2
        right_gradient = (
            2
            * (y[grid.start : grid.end - 1] - y[grid.start + 1 : grid.end])
            / (grid.dx[1:] + grid.dx[:-1])
        )
        right_flux = right_alpha * right_J * right_gradient
        y1[grid.start : grid.end - 1] -= (
            right_flux / (grid.dx[:-1] * right_J) * grid.active[:-1] * grid.active[1:]
        )

        if grid.left_boundary is Boundary.PERIODIC:
            left_alpha = (grid.alpha[0] + grid.alpha[-1]) / 2
            left_J = (grid.jacobian[0][0] + grid.jacobian[0][-1]) / 2
            left_gradient = (
                2 * (y[grid.end - 1] - y[grid.start]) / (grid.dx[-1] + grid.dx[0])
            )
            left_flux = left_alpha * left_J * left_gradient
            y1[grid.start] += left_flux / (grid.dx[0] * left_J)
        elif grid.left_boundary is Boundary.CONSTANT:
            y1[grid.start] = grid.left_boundary_value

        if grid.right_boundary is Boundary.PERIODIC:
            right_alpha = (grid.alpha[-1] + grid.alpha[0]) / 2
            right_J = (grid.jacobian[-1][-1] + grid.jacobian[-1][0]) / 2
            right_gradient = (
                2 * (y[grid.end - 1] - y[grid.start]) / (grid.dx[0] + grid.dx[-1])
            )
            right_flux = right_alpha * right_J * right_gradient
            y1[grid.end - 1] -= right_flux / (grid.dx[-1] * right_J)
        elif grid.right_boundary is Boundary.CONSTANT:
            y1[grid.end - 1] = grid.right_boundary_value

    return y1


def diffusion_periodic_split(t, y, alpha, dx, J):

    y1 = np.zeros(len(y))

    for i in range(1, len(y) - 1):
        # LEFT
        alpha_grad = (alpha[i] + alpha[i - 1]) / 2
        J_grad = (J[i][i] + J[i][i - 1]) / 2
        gradient = 2 * (y[i - 1] - y[i]) / (dx[i - 1] + dx[i])
        flux = alpha_grad * J_grad * gradient

        y1[i] += flux / (dx[i] * J_grad)

        # RIGHT
        alpha_grad = (alpha[i] + alpha[i + 1]) / 2
        J_grad = (J[i][i] + J[i][i + 1]) / 2
        gradient = 2 * (y[i] - y[i + 1]) / (dx[i + 1] + dx[i])
        flux = alpha_grad * J_grad * gradient

        y1[i] -= flux / (dx[i] * J_grad)

    # Left boundary
    # LEFT
    alpha_grad = (alpha[0] + alpha[-1]) / 2
    J_grad = (J[0][0] + J[0][-1]) / 2
    gradient = 2 * (y[-1] - y[0]) / (dx[-1] + dx[0])
    flux = alpha_grad * J_grad * gradient

    y1[0] += flux / (dx[0] * J_grad)

    # RIGHT
    alpha_grad = (alpha[0] + alpha[1]) / 2
    J_grad = (J[0][0] + J[0][1]) / 2
    gradient = 2 * (y[0] - y[1]) / (dx[1] + dx[0])
    flux = alpha_grad * J_grad * gradient

    y1[0] -= flux / (dx[0] * J_grad)

    # Right boundary
    # LEFT
    alpha_grad = (alpha[-1] + alpha[-2]) / 2
    J_grad = (J[-1][-1] + J[-1][-2]) / 2
    gradient = 2 * (y[-2] - y[-1]) / (dx[-2] + dx[-1])
    flux = alpha_grad * J_grad * gradient

    y1[-1] += flux / (dx[-1] * J_grad)

    # RIGHT
    alpha_grad = (alpha[-1] + alpha[0]) / 2
    J_grad = (J[-1][-1] + J[-1][0]) / 2
    gradient = 2 * (y[-1] - y[0]) / (dx[0] + dx[-1])
    flux = alpha_grad * J_grad * gradient

    y1[-1] -= flux / (dx[-1] * J_grad)

    return y1


def calc_jacobian(num_cells, alpha, dx):
    jec = np.zeros((num_cells, num_cells))

    for i in range(1, num_cells - 1):
        jec[i, i - 1] = alpha[i] / dx[i] ** 2
        jec[i, i] = -2 * alpha[i] / dx[i] ** 2
        jec[i, i + 1] = alpha[i] / dx[i] ** 2

    # Left boundary
    jec[0, -1] = jec[0, 1] = alpha[0] / dx[0] ** 2
    jec[0, 0] = -2 * alpha[0] / dx[0] ** 2
    # Right boundary
    jec[-1, -2] = jec[-1, 0] = alpha[-1] / dx[-1] ** 2
    jec[-1, -1] = -2 * alpha[-1] / dx[-1] ** 2

    return jec


MODEL_LIST = {
    "diff_f": diffusion_fixed,
    "diff_p": diffusion_periodic,
    "diff_p_split": diffusion_periodic_split,
    "diff_chimaera": diffusion_chimaera,
}

Submesh = namedtuple("Submesh", "start end dx alpha")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Multiblock TestBench - MSc Fusion Energy project"
    )
    parser.add_argument("model", help="The model to solve for", choices=MODEL_LIST)
    parser.add_argument(
        "-a",
        "--anim",
        help="Show an animated plot of the temperature over time",
        action="store_true",
    )
    parser.add_argument(
        "-i", "--imshow", help="Show 2D plot of position vs time", action="store_true"
    )
    parser.add_argument(
        "-l",
        "--line",
        help="Show line plot with three different times",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--scatter",
        help="Show scatter plot at two time points",
        action="store_true",
    )
    parser.add_argument(
        "--time",
        help="The time to solve up to. Default is 3000",
        type=int,
        default=3000,
    )
    parser.add_argument(
        "--dx",
        help="The cell width in meters. Default is 0.01m. The split model will use this as the larger width",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--solver",
        help="The solver method solve_ivp should use. Default is RK45.",
        default="RK45",
    )
    parser.add_argument(
        "--file",
        help="Path to a grid description toml file, used by diff_chimaera model.",
        type=Path,
    )
    args = parser.parse_args()

    if args.model == "diff_chimaera":
        if args.file:
            with args.file.open(mode="r") as grid_descrip_file:
                chimaera_grid_descrip = toml.load(grid_descrip_file)

            grid_collection = ChimaeraGrid()
            for grid_name, grid in chimaera_grid_descrip["grids"].items():
                new_grid = Grid(
                    grid_name,
                    grid["left_pos"],
                    grid["right_pos"],
                    grid["dx"],
                    grid["alpha"],
                    left_boundary=Boundary[grid["left_boundary"]],
                    right_boundary=Boundary[grid["right_boundary"]],
                )
                grid_collection.add_grid(
                    new_grid,
                    interface_width=grid["interface_width"],
                    num_fringe_cells=grid["num_fringe_cells"],
                )
            grid_collection.ready(
                starting_condition=chimaera_grid_descrip["starting_condition"]
            )
            print(grid_collection)
            grid_collection.solve(
                chimaera_grid_descrip["time_span"], chimaera_grid_descrip["solver"]
            )
            if args.scatter:
                grid_collection.scatter_plot(0.001)

        else:
            base = Grid(
                "base",
                0,
                1,
                args.dx,
                alpha=1,
                left_boundary=Boundary.PERIODIC,
                right_boundary=Boundary.PERIODIC,
            )

            over = Grid("right overlap", 0.45, 0.6, args.dx / 2)
            left_over = Grid("left overlap", 0.2, 0.35, args.dx / 4)

            grid_collection = ChimaeraGrid()
            grid_collection.add_grid(base)
            grid_collection.add_grid(over)
            grid_collection.add_grid(left_over, num_fringe_cells=1)
            # grid_collection.add_pos_value_starting_condition(0.4, 1000)
            grid_collection.ready()
            print(grid_collection)
            grid_collection.solve((0, args.time), args.solver)
            print(grid_collection)
            # print(grid_collection.solver)
            # grid_collection.scatter_plot()
            grid_collection.print_energy_check()
            if args.scatter:
                grid_collection.scatter_plot(0.001)
            # print(grid_collection.grids[)

    else:
        length = 1  # meters

        if args.model == "diff_p_split":
            split_pos = 0.5  # meters
            num_left_cells = int(split_pos / args.dx)
            num_right_cells = int((length - split_pos) / (args.dx / 2))

            # rough description of the mesh made of submeshes with different cell widths
            mesh_descrip = [
                Submesh(1, num_left_cells, args.dx, 1),
                Submesh(
                    num_left_cells, num_left_cells + num_right_cells - 1, args.dx / 2, 1
                ),
                #  Submesh(num_left_cells, num_left_cells + num_right_cells - 1, args.dx),
            ]
            num_cells = num_left_cells + num_right_cells
        else:
            num_cells = int(length / args.dx)
            mesh_descrip = [Submesh(1, num_cells - 1, args.dx, 1)]

        dx_array = np.empty(num_cells)
        alpha_array = np.empty(num_cells)
        for mesh in mesh_descrip:
            dx_array[mesh.start : mesh.end] = mesh.dx
            alpha_array[mesh.start : mesh.end] = mesh.alpha
        dx_array[0] = mesh_descrip[0].dx
        dx_array[-1] = mesh_descrip[-1].dx
        alpha_array[0] = mesh_descrip[0].alpha
        alpha_array[-1] = mesh_descrip[-1].alpha
        y = np.zeros(num_cells)
        y[40] = 1000
        # alpha = 1

        J = calc_jacobian(num_cells, alpha_array, dx_array)

        solver_start_time = time.perf_counter()
        solver = solve_ivp(
            MODEL_LIST[args.model],
            (0, args.time),
            y,
            args=(alpha_array, dx_array, J),
            method=args.solver,
            # jac=J,
        )
        solver_elapsed_time = time.perf_counter() - solver_start_time

        # Calculate the total energy for each timestep and the position of each cell
        if args.model == "diff_p_split":
            energy_array = np.zeros(len(solver.t))
            for m in mesh_descrip:
                energy_array[:] += np.sum(solver.y[m.start : m.end, :], axis=0) * m.dx
            # The boundary cells are currently not included in the mesh description
            energy_array[:] += solver.y[0, :] * mesh_descrip[0].dx
            energy_array[:] += solver.y[-1, :] * mesh_descrip[-1].dx

            cell_pos = np.zeros(num_cells)
            cell_pos[0 : mesh_descrip[0].end] = np.linspace(
                args.dx / 2, split_pos - (mesh_descrip[0].dx / 2), num_left_cells
            )
            cell_pos[mesh_descrip[1].start : mesh_descrip[1].end + 1] = np.linspace(
                split_pos + (mesh_descrip[1].dx / 2),
                length - (mesh_descrip[1].dx / 2),
                num_right_cells,
            )
        else:
            energy_array = np.sum(solver.y, axis=0) * args.dx
            cell_pos = np.linspace(args.dx / 2, length - (args.dx / 2), num_cells)

        print(solver)
        print(f"Elapsed time for solver was {solver_elapsed_time} seconds")
        print("Start energy", energy_array[0], "end energy", energy_array[-1])
        print("Energy diff", energy_array[-1] - energy_array[0])

        # Start of plots

        if args.imshow:
            # 2D plot of the cell index against time
            plt.imshow(solver.y, cmap="inferno", aspect="auto", interpolation="none")

            plt.colorbar()

            plt.xlabel("Time")
            plt.ylabel("Cell index")  # Cell index arn't corrected to positions yet
            plt.grid(None)
            plt.show()

        if args.anim:
            # Plot of the cell index streched to 2D and animated per timestep
            fig, ax = plt.subplots()
            frames = []
            step = 50 if len(solver.t) > 1000 else 1
            for step in range(0, len(solver.t), step):
                frames.append(
                    [
                        ax.imshow(
                            np.expand_dims(solver.y[:, step], axis=0),
                            cmap="inferno",
                            animated=True,
                            aspect="auto",
                        )
                    ]
                )
            ani = animation.ArtistAnimation(
                fig, frames, interval=20, blit=True, repeat_delay=5000
            )

            # ani.save("out.gif", writer='imagemagick')

            plt.show()

        if args.line:
            # Line plot of the temperture against cell position for three points in time
            plt.plot(
                cell_pos,
                solver.y[:, 10],
                label=f"$t={solver.t[10]:.3e}$",
                linestyle="--",
            )
            plt.plot(cell_pos, solver.y[:, 40], label=f"$t={solver.t[40]:.3e}$")
            plt.plot(
                cell_pos,
                solver.y[:, -1],
                label=f"$t={solver.t[-1]:.3e}$",
                linestyle=":",
            )

            plt.legend()
            plt.xlabel("Position")
            plt.ylabel("Temperture")
            plt.show()

        if args.scatter:
            plt.scatter(
                cell_pos, solver.y[:, 10], label=f"$t={solver.t[10]:.3e}$", marker="D"
            )
            plt.scatter(cell_pos, solver.y[:, 40], label=f"$t={solver.t[40]:.3e}$")
            plt.scatter(
                cell_pos, solver.y[:, -1], label=f"$t={solver.t[-1]:.3e}$", marker="x"
            )
            plt.legend()
            plt.xlabel("Position")
            plt.ylabel("Temperture")
            if args.model == "diff_p_split":
                plt.axvline(split_pos, 0, 1)
            plt.show()
