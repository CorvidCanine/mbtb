import mbtb
import argparse
import toml
import numpy as np
from pathlib import Path


class Parameter:
    def __init__(self, name, descrip):
        """Initialisation method for the Parameter class.

        descrip should contain a dictionary with a 'type',
        of either 'range', 'array' or 'ratio'.

        'range' creates a evenly spaced array from a
        'start' value, 'end' value and the 'number' of steps
        to take between those values.

        If the type is 'array', then the 'array' entry should
        be a list of values to be run.

        If the type is 'ratio' then a 'grid' should be specified
        to be multiplied by 'factor' to produce a scaled copy of
        the values to be tested in 'grid'.

        The number of values to run for this parameter must be the
        same for all grids, but dosn't need to be the same as other
        parameters being sweeped.

        Parameters
        ----------
        name : str
            The parameter name
        descrip : dict
            The type of parameter sweep and any needed
            values for it

        Raises
        ------
        ValueError
            Will raise an exception if the type is not recognised or
            the lengths for different grids dosn't match
        """
        self._index = 0
        self.name = name
        self.length = None

        for grid_name, para_grid in descrip.items():

            if para_grid["type"] == "range":
                para_grid["array"] = np.linspace(
                    para_grid["start"], para_grid["end"], para_grid["number"]
                )
            elif para_grid["type"] == "logrange":
                para_grid["array"] = np.logspace(
                    para_grid["start"], para_grid["end"], para_grid["number"]
                )
            elif para_grid["type"] == "array":
                raise NotImplementedError()
            elif para_grid["type"] == "ratio":
                para_grid["array"] = (
                    descrip[para_grid["grid"]]["array"] * para_grid["factor"]
                )
            else:
                raise ValueError(
                    f"Parameter type {para_grid['type']} for parameter {self.name} is not recognised"
                )

            if self.length is None:
                self.length = len(para_grid["array"])
            elif self.length != len(para_grid["array"]):
                raise ValueError(
                    f"The length of the parameter, {name}, must be the same for all grids.",
                    f"Length so far was {self.length}, length for grid {grid_name} is {len(para_grid['array'])}",
                )
            self.descrip = descrip

    def __iter__(self):
        return self

    def __next__(self):

        if self._index >= self.length:
            raise StopIteration

        grid_value_dict = {}
        for grid_name, para_grid in self.descrip.items():
            grid_value_dict[grid_name] = para_grid["array"][self._index]

        self._index += 1

        return grid_value_dict


def run(chimaera_grid_descrip):

    chimaera_grid = mbtb.ChimaeraGrid(
        name=chimaera_grid_descrip["name"],
        description=chimaera_grid_descrip["description"],
    )

    try:
        for grid_name, grid in chimaera_grid_descrip["grids"].items():
            new_grid = mbtb.Grid(
                grid_name,
                grid["left_pos"],
                grid["right_pos"],
                grid["dx"],
                grid["alpha"],
                left_boundary=mbtb.Boundary[grid["left_boundary"]],
                right_boundary=mbtb.Boundary[grid["right_boundary"]],
            )
            if mbtb.Boundary[grid["left_boundary"]] == mbtb.Boundary.CONSTANT:
                new_grid.set_constant_boundary_values(left=grid["left_boundary_value"])
            if mbtb.Boundary[grid["right_boundary"]] == mbtb.Boundary.CONSTANT:
                new_grid.set_constant_boundary_values(
                    right=grid["right_boundary_value"]
                )

            chimaera_grid.add_grid(
                new_grid,
                interface_width=grid["interface_width"],
                num_fringe_cells=grid["num_fringe_cells"],
            )
    except mbtb.OverlapError as e:
        if args.ignore_overlap_errors:
            # If ignoring overlap errors, return to be saved so we
            # know the failed grid was attempted.
            # The 'was_readied' and 'solve_attempted' flags will be
            # false in the chimaera_grid. The overlap that failed
            # will have its 'failed_to_create' flag set to True.
            return chimaera_grid
        else:
            # If we're not ignoring, re-raise the error
            raise e

    chimaera_grid.ready(starting_condition=chimaera_grid_descrip["starting_condition"])
    try:
        chimaera_grid.solve(
            chimaera_grid_descrip["time_span"],
            chimaera_grid_descrip["solver"],
        )
    except ValueError as e:
        # TODO Temporarily excepting ValueErrors thrown by interp_func
        # Need to fix this, as overlaps that arn't going to work are meant
        # to be caught before that point.
        if args.ignore_overlap_errors:
            print("Value Error")
            return chimaera_grid
        else:
            raise e

    return chimaera_grid


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Multi-Block Test Bench - Sweep",
        description="Perform parameter sweeps of Multi-Block Test Bench grids",
    )
    parser.add_argument(
        "sweep",
        help="Path to a sweep description toml file",
        type=Path,
    )
    parser.add_argument(
        "grid",
        help="Path to a grid description toml file",
        type=Path,
    )
    parser.add_argument(
        "--save",
        help="Path to save results to, a subdirectory will be created from"
        + " the name of the sweep. Default is 'out'.",
        type=Path,
        default=Path("out"),
    )
    parser.add_argument(
        "-i",
        help="Ignore overlap errors. Save the failed run and continue",
        action="store_true",
        dest="ignore_overlap_errors",
    )
    parser.add_argument(
        "-f",
        help="Force use of the sweep if there are grids missing. Missing grids will be ignored.",
        action="store_true",
        dest="ignore_missing_grids",
    )
    parser.add_argument(
        "-j",
        help="Save runs to JSON instead of HDF5",
        action="store_true",
        dest="save_to_json",
    )
    args = parser.parse_args()

    if args.save_to_json:
        save_format = "json"
    else:
        save_format = "h5"

    with args.sweep.open(mode="r") as sweep_file:
        sweep_descrip = toml.load(sweep_file)

    with args.grid.open(mode="r") as grid_descrip_file:
        chimaera_grid_descrip = toml.load(grid_descrip_file)

    if "solvers" not in sweep_descrip:
        # If no solver array was specified, use the solver in the grid description
        solver_list = [chimaera_grid_descrip["solver"]]
    else:
        solver_list = sweep_descrip["solvers"]

    save_path = args.save / sweep_descrip["name"] / chimaera_grid_descrip["name"]
    file_counter = 0

    for solver in solver_list:
        # Loop though all the parameters that are to be sweeped
        for descrip_para_i_name, descrip_para_i in sweep_descrip["parameters"].items():

            para_i = Parameter(descrip_para_i_name, descrip_para_i)

            if len(sweep_descrip["parameters"]) == 1:
                # Loop though each value to be tested for the parameter
                for para_i_values in para_i:
                    # Need to make a copy of the grid description so we don't
                    # alter the original for future runs
                    descrip_to_run = chimaera_grid_descrip.copy()
                    # As we just made a copy, the solver needs to be set every loop
                    descrip_to_run["solver"] = solver

                    for grid_name, para_value in para_i_values.items():
                        try:
                            descrip_to_run["grids"][grid_name][para_i.name] = para_value
                        except KeyError as e:
                            if not args.ignore_missing_grids:
                                # If the flag to ignore grids in the sweep description file that
                                # are missing is not set, re-raise the KeyError
                                raise KeyError(
                                    f"The grid '{grid_name}' is not in this chimaera grid. The -f flag can suppress this message."
                                )

                    chimaera_grid = run(descrip_to_run)
                    chimaera_grid.save(
                        save_path / str(file_counter),
                        sweep_name=sweep_descrip["name"],
                        file_format=save_format,
                    )

                    if chimaera_grid.is_solved:
                        print(
                            f"Run {file_counter} completed in {chimaera_grid.solver_elapsed_time:6.1f}s"
                        )
                    else:
                        print(f"Run {file_counter} was not solved")

                    file_counter += 1

            else:
                # If there is more than one parameter being sweeped, a second
                # nested loop is needed
                for descrip_para_j_name, descrip_para_j in sweep_descrip[
                    "parameters"
                ].items():

                    if descrip_para_i_name == descrip_para_j_name:
                        continue

                    para_j = Parameter(descrip_para_j_name, descrip_para_j)

                    for para_i_values in para_i:
                        for para_j_values in para_j:
                            descrip_to_run = chimaera_grid_descrip.copy()
                            descrip_to_run["solver"] = solver

                            for grid_name, para_value in para_i_values.items():
                                try:
                                    descrip_to_run["grids"][grid_name][
                                        para_i.name
                                    ] = para_value
                                except KeyError as e:
                                    if not args.ignore_missing_grids:
                                        # If the flag to ignore grids in the sweep description file that
                                        # are missing is not set, re-raise the KeyError
                                        raise e

                            for grid_name, para_value in para_j_values.items():
                                try:
                                    descrip_to_run["grids"][grid_name][
                                        para_j.name
                                    ] = para_value
                                except KeyError as e:
                                    if not args.ignore_missing_grids:
                                        raise e

                            chimaera_grid = run(descrip_to_run)
                            chimaera_grid.save(
                                save_path / str(file_counter),
                                sweep_name=sweep_descrip["name"],
                                file_format=save_format,
                            )

                            if chimaera_grid.is_solved:
                                print(
                                    f"Run {file_counter} completed in {chimaera_grid.solver_elapsed_time:6.1f}s"
                                )
                            else:
                                print(f"Run {file_counter} was not solved")

                            file_counter += 1
