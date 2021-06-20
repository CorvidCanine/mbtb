import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import style
from matplotlib import animation
from collections import namedtuple

try:
    # Load custom matplotlib style if it's avalible
    style.use(["nord-base-small", "corvid-light"])
except OSError:
    pass


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
        y1[i] = alpha * (y[i - 1] - 2 * y[i] + y[i + 1]) / dx ** 2

    # Apply periodic boundary condition
    y1[0] = alpha * (y[-1] - 2 * y[0] + y[1]) / dx ** 2
    y1[-1] = alpha * (y[-2] - 2 * y[-1] + y[0]) / dx ** 2

    return y1


def diffusion_periodic_split(t, y, alpha, mesh):

    y1 = np.zeros(len(y))

    # Loop though "submesh" descriptions
    for m in mesh:
        # Loop though the cells for that submesh
        for i in range(m.start, m.end):
            y1[i] = alpha * (y[i - 1] - 2 * y[i] + y[i + 1]) / m.dx ** 2

    # Apply periodic boundary condition - dx is currently hardcoded here
    y1[0] = alpha * (y[-1] - 2 * y[0] + y[1]) / mesh[0].dx ** 2
    y1[-1] = alpha * (y[-2] - 2 * y[-1] + y[0]) / mesh[-1].dx ** 2

    return y1


MODEL_LIST = {
    "diff_f": diffusion_fixed,
    "diff_p": diffusion_periodic,
    "diff_p_split": diffusion_periodic_split,
}

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
    args = parser.parse_args()

    length = 1  # meters

    if args.model == "diff_p_split":

        split_pos = 0.5  # meters
        num_left_cells = int(split_pos / args.dx)
        num_right_cells = int((length - split_pos) / (args.dx / 2))

        Submesh = namedtuple("Submesh", "start end dx")
        # rough description of the mesh made of submeshes with different cell widths
        mesh_descrip = [
            Submesh(1, num_left_cells, args.dx),
            Submesh(num_left_cells, num_left_cells + num_right_cells - 1, args.dx / 2),
        ]
        num_cells = num_left_cells + num_right_cells
    else:
        num_cells = int(length / args.dx)
        mesh_descrip = args.dx

    y = np.zeros(num_cells)
    y[40] = 1000
    alpha = 1

    solver_start_time = time.perf_counter()
    solver = solve_ivp(
        MODEL_LIST[args.model],
        (0, args.time),
        y,
        args=(alpha, mesh_descrip),
        method=args.solver,
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
            cell_pos, solver.y[:, 10], label=f"$t={solver.t[10]:.3e}$", linestyle="--"
        )
        plt.plot(cell_pos, solver.y[:, 40], label=f"$t={solver.t[40]:.3e}$")
        plt.plot(
            cell_pos, solver.y[:, -1], label=f"$t={solver.t[-1]:.3e}$", linestyle=":"
        )

        plt.legend()
        plt.xlabel("Position")
        plt.ylabel("Temperture")
        plt.show()

    if args.scatter:
        plt.scatter(cell_pos, solver.y[:, 10], label=f"$t={solver.t[10]:.3e}$", marker="D")
        plt.scatter(cell_pos, solver.y[:, 40], label=f"$t={solver.t[40]:.3e}$")
        plt.legend()
        plt.xlabel("Position")
        plt.ylabel("Temperture")
        plt.show()
