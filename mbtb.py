import time
import warnings
import json
import subprocess
import platform
import hdf5storage
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib import style
from enum import Enum, auto
from datetime import datetime

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


class OverlapError(Exception):
    """Exception raised when an overlap can not be created"""

    pass


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
        OverlapError
            If num_fringe_cells is less than 1 or the interface width is
            too short for the cell widths to create an overlap an exception
            is raised
        """
        self.failed_to_create = False
        self.lower_grid = lower_grid
        self.over_grid = over_grid

        if num_fringe_cells < 1:
            raise OverlapError("The number of overlap cells must be at least 1")

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

        if (
            len(self.lower_left_interp_positions) < 2
            or self.over_left_fringe_positions > self.lower_left_interp_positions[-1]
        ):
            self.failed_to_create = True
            raise OverlapError(
                f"Unable to create overlap for '{self.over_grid.name}' on '{self.lower_grid.name}',"
                + " insufficient interface width on left side"
            )

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

        if (
            len(self.lower_right_interp_positions) < 2
            or self.over_right_fringe_positions < self.lower_right_interp_positions[0]
        ):
            self.failed_to_create = True
            raise OverlapError(
                f"Unable to create overlap for '{self.over_grid.name}' on '{self.lower_grid.name}',"
                + " insufficient interface width on right side"
            )

    def __str__(self):
        return f"Overlap of grid {self.over_grid.index} on lower grid {self.lower_grid.index} with {self.num_fringe_cells} fringe cells"

    def to_dict(self):
        """Returns a dictionary containing all the important attributes in the object

        Returns
        -------
        dict
            The dict containing all attributes
        """
        overlap_dict = {}
        # Save the indexes to the grids, not copies of them
        overlap_dict["lower_grid_index"] = self.lower_grid.index
        overlap_dict["over_grid_index"] = self.over_grid.index
        overlap_dict["failed_to_create"] = self.failed_to_create
        overlap_dict["num_fringe_cells"] = self.num_fringe_cells
        overlap_dict["interface_width"] = self.interface_width
        overlap_dict["lower_left_interp_range"] = self.lower_left_interp_range
        overlap_dict["lower_left_interp_positions"] = self.lower_left_interp_positions
        overlap_dict["lower_right_interp_range"] = self.lower_right_interp_range
        overlap_dict["lower_right_interp_positions"] = self.lower_right_interp_positions
        overlap_dict["over_left_interp_range"] = self.over_left_interp_range
        overlap_dict["over_left_interp_positions"] = self.over_left_interp_positions
        overlap_dict["over_right_interp_range"] = self.over_right_interp_range
        overlap_dict["over_right_interp_positions"] = self.over_right_interp_positions

        return overlap_dict


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
    name : str
        A name for this Chimaera grid
    description : str
        A short description of this Chimaera grid
    """

    def __init__(self, name=None, description=None):
        """Initialisation method for ChimaeraGrid

        Parameters
        ----------
        name : str, optional
            A name for this chimaera grid, by default None
        description : str, optional
            A short description for this chimaera grid, by default None
        """

        self.name = name
        self.description = description
        self.grids = []
        self.overlaps = []
        self.starting_condition = None
        self.cell_positions = np.array([])
        self.active_y = np.array([])
        self.total_num_cells = 0
        self.result = None
        self.solver_elapsed_time = None
        self.solver_time_span = None
        self.solver_method = None
        self.total_energy = None

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
        interface_width : float
            The width of the interface region, will be passed to
            a new Overlap object.
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
            raise Exception("At least one grid needs to be registered before readying")

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
            starting_condition = {"type": "preset", "starting_array": starting_y}
        elif starting_condition["type"] == "gaussian":
            starting_y[self.active_y] = gaussian(
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

        self.starting_condition = starting_condition
        self.starting_y = starting_y
        self.is_ready = True

    def solve(self, solver_time_span, solver_method, complete_msg=False):
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
        complete_msg : bool
            Prints a short message on completion when True, default False

        Raises
        ------
        Exception
            If the method `ready` hasn't been run an exception will be raised
        """

        if not self.is_ready:
            raise Exception("The ready method must be run before solving")

        # Store the arguments and solver method used
        self.solver_time_span = solver_time_span
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

        if complete_msg:
            print(f"Ran for {self.solver_elapsed_time:6.1f}s. {solver.message}")

        self.result = solver
        self.is_solved = True
        self._calculate_uniform_and_continuous_solutions()

    def smallest_cell_width(self):
        smallest_cell_width = None
        for grid in self.grids:
            smallest_dx_grid = grid.dx.min()
            if smallest_cell_width is None:
                smallest_cell_width = smallest_dx_grid
            elif smallest_cell_width > smallest_dx_grid:
                smallest_cell_width = smallest_dx_grid

        return smallest_cell_width

    def _calculate_uniform_and_continuous_solutions(self):
        smallest_dx = self.smallest_cell_width()

        number_cells = int(1 / smallest_dx)
        self.uniform_solution_positions = np.linspace(
            0 + (smallest_dx) / 2, 1 - (smallest_dx) / 2, number_cells
        )
        self.uniform_solution = np.empty((number_cells, len(self.result.t)))
        self.uniform_solution[:] = np.NaN
        self.cont_solution_positions = np.empty(0)
        for grid in self.grids:
            active_bool = grid.active.astype(bool)
            try:
                self.cont_solution = np.append(
                    self.cont_solution, grid.solution[active_bool, :], axis=0
                )
            except AttributeError:
                self.cont_solution = grid.solution[active_bool, :]

            self.cont_solution_positions = np.append(
                self.cont_solution_positions, grid.cell_positions[active_bool]
            )

            interp_func = interp1d(
                grid.cell_positions,
                grid.solution,
                axis=0,
                bounds_error=False,
                assume_sorted=True,
            )
            interped_grid = interp_func(self.uniform_solution_positions)
            self.uniform_solution[~np.isnan(interped_grid)] = interped_grid[
                ~np.isnan(interped_grid)
            ]

        sorted_cont_order = np.argsort(self.cont_solution_positions)
        self.cont_solution_positions = np.take_along_axis(
            self.cont_solution_positions, sorted_cont_order, 0
        )
        self.cont_solution = np.take(self.cont_solution, sorted_cont_order, 0)

        self.uniform_cell_width = smallest_dx

    def print_energy_check(self):
        """Print the energy difference for each grid and the total to terminal"""

        if not self.is_solved:
            print("This grid collection has not been solved yet")
            return

        print(f"{'Grid':^20}  Start energy   End energy   Energy difference")
        for grid in self.grids:
            print(
                f"{grid.name:^20}  {grid.energy[0]:10.4f}     {grid.energy[-1]:10.4f}    {grid.energy[-1] - grid.energy[0]:10.4f}"
            )
        print(
            f"{'Total':^20}  {self.total_energy[0]:10.4f}     {self.total_energy[-1]:10.4f}    {self.total_energy[-1] - self.total_energy[0]:10.4f}"
        )

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

    def to_dict(self):
        """Returns a dictionary containing all the important attributes in the object

        Returns
        -------
        dict
            The dict containing all attributes
        """
        chimaera_grid_dict = {}
        chimaera_grid_dict["solved_attempted"] = self.is_solved
        if self.is_solved:
            chimaera_grid_dict["uniform_solution"] = self.uniform_solution
            chimaera_grid_dict["uniform_cell_width"] = self.uniform_cell_width
            chimaera_grid_dict["cont_solution"] = self.cont_solution
            chimaera_grid_dict["chont_solution_pos"] = self.cont_solution_positions
            chimaera_grid_dict["solver_time"] = self.solver_elapsed_time
            chimaera_grid_dict["total_energy"] = self.total_energy
            chimaera_grid_dict[
                "uniform_solution_positions"
            ] = self.uniform_solution_positions
            # The result from solve_ivp is a subclass of OptimizeResult,
            # which is a subclass of dict, so this dosn't change too much
            # but can be serialised
            chimaera_grid_dict["result"] = dict(self.result)

        chimaera_grid_dict["name"] = self.name
        chimaera_grid_dict["description"] = self.description
        chimaera_grid_dict["grids"] = self.grids
        chimaera_grid_dict["overlaps"] = self.overlaps
        chimaera_grid_dict["total_num_cells"] = self.total_num_cells
        chimaera_grid_dict["was_readied"] = self.is_ready
        chimaera_grid_dict["starting_condition"] = self.starting_condition
        chimaera_grid_dict["solver_time_span"] = self.solver_time_span
        chimaera_grid_dict["solver_method"] = self.solver_method

        return chimaera_grid_dict

    def get_header_dict(self):
        """Create a new header dictionary

        Gather some useful infomation to
        add to save files of the object.

        Returns
        -------
        dict
            The header dictionary
        """
        header = {
            "timestamp": datetime.now().isoformat(),
            "git_hash": get_git_hash(),
            "platform": platform.platform(),
            "node": platform.node(),
            "python_version": platform.python_version(),
        }
        return header

    def save(
        self,
        save_path,
        allow_overwrite=False,
        sweep_name=None,
        file_format="h5",
        hdf5_path="/",
    ):
        """Save the Chimaera object to disk.

        Parameters
        ----------
        save_path : Path
            The path of the file to save to
        allow_overwrite : bool, optional
            Should existing files be overwritten?
            By default False
        sweep_name : str, optional
            If part of a sweep, a name for the sweep can
            be added to the file, by default None
        file_format : str, optional
            The file format to save to. Can be either "json"
            or "h5", by default "h5". If save_path has a file extension
            that will be used instead.
        hdf5_path : str, optional
            The path within the hdf5 file to write to, by default "/"

        Raises
        ------
        ValueError
            Will raise an exception is a unrecognized file format is asked for
        """

        dict_to_save = self.get_header_dict()
        dict_to_save["sweep_name"] = sweep_name

        if save_path.suffix == "":
            # If the path dosn't have a file extension, add one
            save_path = save_path.with_suffix("." + file_format)

        # Create the directories/folders if they don't already exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            # Can let the MBTBEncoder serialise this ChimaeraGrid class
            dict_to_save["chimaera_grid"] = self
            with save_path.open(mode="w" if allow_overwrite else "x") as save_file:
                json.dump(dict_to_save, save_file, cls=MBTBEncoder)
        elif save_path.suffix == ".h5":

            # No help from the encoder here, need to go through the grids
            # and overlaps and convert everything into a dictionary.
            # Don't convert any numpy arrays or objects though.
            chimaera_grid_dict = self.to_dict()
            grid_dict_list = list()
            overlap_dict_list = list()
            for grid_obj in self.grids:
                grid_dict_list.append(grid_obj.to_dict())
            for overlap_obj in self.overlaps:
                overlap_dict_list.append(overlap_obj.to_dict())
            chimaera_grid_dict["grids"] = grid_dict_list
            chimaera_grid_dict["overlaps"] = overlap_dict_list

            dict_to_save["chimaera_grid"] = chimaera_grid_dict

            hdf5storage.write(
                dict_to_save,
                path=hdf5_path,
                # The h5py module dosn't like Path objects :(
                filename=save_path.as_posix(),
                store_python_metadata=True,
                truncate_existing=allow_overwrite,
            )
        else:
            raise ValueError(
                f"'{save_path.suffix}' is not a supported format to save to"
            )

    def __str__(self):
        return (
            f"Grid Collection of {len(self.grids)} grids and {len(self.overlaps)} overlaps, "
            + f"has {'' if self.is_solved else 'not '}been solved"
        )


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
    left_boundary_value : float
        If the left boundary is type CONSTANT, it has this value
    right_boundary_value : float
        If the right boundary is type CONSTANT, it has this value
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
        self.left_boundary_value = None
        self.right_boundary_value = None

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

    def to_dict(self):
        """Returns a dictionary containing all the important attributes in the object

        Returns
        -------
        dict
            The dict containing all attributes
        """
        grid_dict = {}
        grid_dict["name"] = self.name
        grid_dict["index"] = self.index
        grid_dict["left_pos"] = self.left_pos
        grid_dict["right_pos"] = self.right_pos
        grid_dict["num_cells"] = self.num_cells
        grid_dict["cell_widths"] = self.dx
        grid_dict["alpha"] = self.alpha
        grid_dict["cell_positions"] = self.cell_positions
        # Save the active array as bools, otherwise it will be
        # saved as floats
        grid_dict["active_cells"] = self.active.astype(bool)
        grid_dict["left_boundary"] = self.left_boundary.name
        grid_dict["right_boundary"] = self.right_boundary.name
        grid_dict["left_boundary_value"] = self.left_boundary_value
        grid_dict["right_boundary_value"] = self.right_boundary_value
        grid_dict["jacobian"] = self.jacobian
        grid_dict["start_index"] = self.start
        grid_dict["end_index"] = self.end
        grid_dict["solution"] = self.solution
        grid_dict["energy"] = self.energy
        return grid_dict


class MBTBEncoder(json.JSONEncoder):
    """Custom JSON encoder

    Converts MBTB classes to dictionaries
    and some numpy objects to python native
    equivalents.
    """

    def default(self, obj):
        if isinstance(obj, ChimaeraGrid):
            return obj.to_dict()
        if isinstance(obj, Grid):
            return obj.to_dict()
        if isinstance(obj, Overlap):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super().default(obj)


def gaussian(positions, height, centre, width, base, time=0):
    """Calculates a gaussian for the passed positions.

    For creating a starting gaussian and for finding
    the solution to a diffusion guassian over time. Works with
    xarrays.

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
    time : float or ndarray, shape(num_cells)
        The time of the gaussian when representing a diffusion
        solution. 1 will be added to the time.

    Returns
    -------
    ndarray, shape(num_cells)
        A starting array of values
    """
    time = time + 1
    return base + height / np.sqrt(time) * np.exp(
            -((positions - centre) ** 2) / (2 * width * width * time)
        )

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


def get_git_hash():
    """Returns the last hash for the mbtb git repo

    Returns
    -------
    str
        The git hash
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")


MODEL_LIST = {
    "diff_f": diffusion_fixed,
    "diff_p": diffusion_periodic,
    "diff_p_split": diffusion_periodic_split,
    "diff_chimaera": diffusion_chimaera,
}
