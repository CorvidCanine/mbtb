import mbtb
import time
import argparse
import toml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import namedtuple
from matplotlib import animation
from scipy.integrate import solve_ivp


Submesh = namedtuple("Submesh", "start end dx alpha")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Multiblock TestBench - MSc Fusion Energy project"
    )
    parser.add_argument("model", help="The model to solve for", choices=mbtb.MODEL_LIST)
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

            grid_collection = mbtb.ChimaeraGrid(
                name=chimaera_grid_descrip["name"],
                description=chimaera_grid_descrip["description"],
            )
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
                grid_collection.scatter_plot(0)
                grid_collection.scatter_plot(0.01)

        else:
            base = mbtb.Grid(
                "base",
                0,
                1,
                args.dx,
                alpha=1,
                left_boundary=mbtb.Boundary.PERIODIC,
                right_boundary=mbtb.Boundary.PERIODIC,
            )

            over = mbtb.Grid("right overlap", 0.45, 0.6, args.dx / 2)
            left_over = mbtb.Grid("left overlap", 0.2, 0.35, args.dx / 4)

            grid_collection = mbtb.ChimaeraGrid()
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

        J = mbtb.calc_jacobian(num_cells, alpha_array, dx_array)

        solver_start_time = time.perf_counter()
        solver = solve_ivp(
            mbtb.MODEL_LIST[args.model],
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
