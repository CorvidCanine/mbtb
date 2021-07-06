import time
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import style
from matplotlib import animation
from collections import namedtuple
from enum import Enum, auto

try:
    # Load custom matplotlib style if it's avalible
    style.use(["nord-base-small", "corvid-light"])
except OSError:
    pass


class Boundary(Enum):
    PERIODIC = auto()
    WALL = auto()
    IGNORE = auto()
    CONSTANT = auto()


class Overlap:
    def __init__(self, lower_grid, over_grid, num_fringe_cells=1):
        self.lower_grid = lower_grid
        self.over_grid = over_grid

        if num_fringe_cells < 1:
            raise ValueError("The number of overlap cells must be at least 1")

        self.num_fringe_cells = num_fringe_cells

        self.lower_grid.cut_hole(
            self.over_grid.left_pos, self.over_grid.right_pos, num_fringe_cells
        )

        left_cut_index = self.lower_grid.position_to_cell(self.over_grid.left_pos)
        self.lower_left_interp_range = (
            left_cut_index,
            left_cut_index + num_fringe_cells + 1,
        )
        self.lower_left_fringe_range = (
            left_cut_index + num_fringe_cells + 1,
            left_cut_index + (2 * num_fringe_cells) + 1,
        )
        self.lower_left_interp_positions = self.lower_grid.cell_positions[
            self.lower_left_interp_range[0] : self.lower_left_interp_range[1]
        ]
        self.over_left_interp_positions = self.over_grid.cell_positions[
            0:num_fringe_cells
        ]

        right_cut_index = self.lower_grid.position_to_cell(self.over_grid.right_pos)
        self.lower_right_interp_range = (
            right_cut_index - num_fringe_cells,
            right_cut_index + 1,
        )
        self.lower_right_fringe_range = (
            right_cut_index - (2 * num_fringe_cells) - 1,
            right_cut_index - num_fringe_cells - 1,
        )
        self.lower_right_interp_positions = self.lower_grid.cell_positions[
            self.lower_right_interp_range[0] : self.lower_right_interp_range[1]
        ]
        self.over_right_interp_positions = self.over_grid.cell_positions[
            -num_fringe_cells:-1
        ]

    def __str__(self):
        return f"Overlap of grid {self.over_grid.index} on lower grid {self.lower_grid.index} with {self.num_fringe_cells} fringe cells"


class GridCollection:
    def __init__(self):
        self.grids = []
        self.overlaps = []

    def add_grid(self, new_grid):
        new_grid.index = len(self.grids)

        for lower_grid in self.grids:

            if lower_grid.does_pos_range_overlap(new_grid.left_pos, new_grid.right_pos):
                self.overlaps.append(Overlap(lower_grid, new_grid))

        self.grids.append(new_grid)

    def __str__(self):
        return f"Grid Collection contains {len(self.grids)} grids and {len(self.overlaps)} overlaps"


class Grid:
    def __init__(
        self,
        left_pos,
        right_pos,
        dx,
        alpha,
        left_boundary=Boundary.WALL,
        right_boundary=Boundary.WALL,
    ):
        self.index = None
        self.left_pos = left_pos
        self.right_pos = right_pos

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

        if isinstance(dx, float):
            self.num_cells = int((self.right_pos - self.left_pos) / dx)
            self.dx = np.full(self.num_cells, dx, dtype=np.float64)
        else:
            self.num_cells = len(dx)
            self.dx = np.array(dx, dype=np.float64)

        if isinstance(alpha, float) or isinstance(alpha, int):
            self.alpha = np.full(self.num_cells, alpha, dtype=np.float64)
        else:
            self.alpha = np.array(alpha, dype=np.float64)

        self.active = np.ones(self.num_cells)
        self.cell_positions = self.dx.cumsum() - self.dx / 2 + self.left_pos

    def __str__(self):
        return f"Grid {self.index}, from {self.left_pos} to {self.right_pos} with {self.num_cells} cells"

    def give_solution(self, solution):
        self.solution = solution
        self.energy = np.sum(solution, axis=0) * self.dx

    def set_constant_boundary_values(self, left=None, right=None):
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

    def position_to_cell(self, pos):
        return np.abs(self.cell_positions - pos).argmin()

    def left_right_cell_pos(self):
        return self.cell_positions[0], self.cell_positions[-1]

    def cut_hole(self, left_cut_pos, right_cut_pos, num_fringe_cells):

        left_cut_index = (
            self.position_to_cell(left_cut_pos) + (2 * num_fringe_cells) + 1
        )
        right_cut_index = (
            self.position_to_cell(right_cut_pos) - (2 * num_fringe_cells) - 1
        )

        # If the cut positions don't overlap the grid at all, we don't want to incorrectly cut an end cell
        if (
            left_cut_index + right_cut_index > 0
            and left_cut_index + right_cut_index < (2 * (self.num_cells - 1))
        ):
            self.active[left_cut_index : right_cut_index + 1] = 0

        return self

    def does_pos_range_overlap(self, leftmost_pos, rightmost_pos):
        return leftmost_pos <= self.right_pos and rightmost_pos >= self.left_pos


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


def diffusion_chimaera(t, y, grids, J):

    y1 = np.zeros(len(y))

    for grid in grids:

        left_alpha = (grid.alpha[1:] + grid.alpha[:-1]) / 2
        left_J = (np.diagonal(J)[1:] + np.diagonal(J, offset=-1)) / 2
        left_gradient = (
            2
            * (y[grid.left : grid.right - 1] - y[grid.left + 1 : grid.right])
            / (grid.dx[:-1] + grid.dx[1:])
        )
        left_flux = left_alpha * left_J * left_gradient

        y1[grid.left + 1 : grid.right] += (
            left_flux / (grid.dx[1:] * left_J) * grid.active[1:] * grid.active[:-1]
        )

        # y1[grid.left + 1 : grid.right] += np.divide(
        #     left_flux,
        #     grid.dx[1:] * left_J,
        #     out=np.zeros_like(left_flux),
        #     where=(grid.active[1:] & (grid.active[:-1]),
        # )

        right_alpha = (grid.alpha[:-1] + grid.alpha[1:]) / 2
        right_J = (np.diagonal(J)[:-1] + np.diagonal(J, offset=1)) / 2
        right_gradient = (
            2
            * (y[grid.left : grid.right - 1] - y[grid.left + 1 : grid.right])
            / (grid.dx[1:] + grid.dx[:-1])
        )
        right_flux = right_alpha * right_J * right_gradient
        y1[grid.left : grid.right - 1] -= (
            right_flux / (grid.dx[:-1] * right_J) * grid.active[:-1] * grid.active[1:]
        )

        # y1[grid.left : grid.right - 1] -= np.divide(
        #     right_flux,
        #     grid.dx[:-1] * right_J,
        #     out=np.zeros_like(right_flux),
        #     where=((grid.active[:-1] != 0)),
        #     # where=(right_J!=0),
        # )

        if grid.leftboundary.type == "periodic":
            left_alpha = (grid.alpha[0] + grid.alpha[-1]) / 2
            left_J = (J[0][0] + J[0][-1]) / 2
            left_gradient = (
                2 * (y[grid.right - 1] - y[grid.left]) / (grid.dx[-1] + grid.dx[0])
            )
            left_flux = left_alpha * left_J * left_gradient
            y1[grid.left] += left_flux / (grid.dx[0] * left_J)
        elif grid.leftboundary.type == "constant":
            y1[grid.left] = grid.leftboundary.value

        if grid.rightboundary.type == "periodic":
            right_alpha = (grid.alpha[-1] + grid.alpha[0]) / 2
            right_J = (J[-1][-1] + J[-1][0]) / 2
            right_gradient = (
                2 * (y[grid.right - 1] - y[grid.left]) / (grid.dx[0] + grid.dx[-1])
            )
            right_flux = right_alpha * right_J * right_gradient
            y1[grid.right - 1] -= right_flux / (grid.dx[-1] * right_J)
        elif grid.leftboundary.type == "constant":
            y1[grid.right - 1] = grid.rightboundary.value

    # print("y", y[grids[0].active == 0])
    # print("y1", y1[grids[0].active == 0])

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


def calc_jacobian_chimaera(num_cells, grids):
    jec = np.zeros((num_cells, num_cells))

    for grid in grids:

        for i in range(grid.left + 1, grid.right - 1):
            # if grid.active[i - 1] == 1:
            left_alpha_grad = (grid.alpha[i] + grid.alpha[i - 1]) / 2
            jec[i, i - 1] = left_alpha_grad / ((grid.dx[i - 1] + grid.dx[i]) / 2) ** 2
            # else:
            #     jec[i, i - 1] = 1

            # if grid.active[i + 1] == 1:
            right_alpha_grad = (grid.alpha[i] + grid.alpha[i + 1]) / 2
            jec[i, i + 1] = right_alpha_grad / (
                ((grid.dx[i + 1] + grid.dx[i]) / 2) ** 2
            )
            # else:
            #     jec[i, i + 1] = 1

            # if grid.active[i] == 1:
            jec[i, i] = -(jec[i, i - 1] + jec[i, i + 1])
            # else:
            #     jec[i, i] = 1

        if grid.leftboundary.type == "periodic":
            left_alpha_grad = (grid.alpha[grid.left] + grid.alpha[grid.right - 1]) / 2
            jec[grid.left, grid.right - 1] = (
                left_alpha_grad
                / ((grid.dx[grid.right - 1] + grid.dx[grid.left]) / 2) ** 2
            )
            right_alpha_grad = (grid.alpha[grid.left] + grid.alpha[grid.left + 1]) / 2
            jec[grid.left, grid.left + 1] = (
                right_alpha_grad
                / ((grid.dx[grid.left + 1] + grid.dx[grid.left]) / 2) ** 2
            )
            jec[grid.left, grid.left] = -(
                jec[grid.left, grid.right - 1] + jec[grid.left, grid.left + 1]
            )

        if grid.rightboundary.type == "periodic":
            left_alpha_grad = (
                grid.alpha[grid.right - 1] + grid.alpha[grid.right - 2]
            ) / 2
            jec[grid.right - 1, grid.right - 2] = (
                left_alpha_grad
                / ((grid.dx[grid.right - 2] + grid.dx[grid.right - 1]) / 2) ** 2
            )
            right_alpha_grad = (grid.alpha[grid.right - 1] + grid.alpha[grid.left]) / 2
            jec[grid.right - 1, grid.left] = (
                right_alpha_grad
                / ((grid.dx[grid.left] + grid.dx[grid.right - 1]) / 2) ** 2
            )
            jec[grid.right - 1, grid.right - 1] = -(
                jec[grid.right - 1, grid.right - 2] + jec[grid.right - 1, grid.left]
            )

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
    args = parser.parse_args()

    length = 1  # meters

    if args.model == "diff_p_split" or args.model == "diff_chimaera":
        # TODO Temporarily runs for chimaera as well
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

    base = Grid(
        0,
        1,
        args.dx,
        1,
        left_boundary=Boundary.PERIODIC,
        right_boundary=Boundary.PERIODIC,
    )
    over = Grid(0.5, 0.6, args.dx / 2, 1)

    grid_collection = GridCollection()
    grid_collection.add_grid(base)
    grid_collection.add_grid(over)

    print(grid_collection)

    # if args.model == "diff_p_split":
    J = calc_jacobian(num_cells, alpha_array, dx_array)

    if args.model == "diff_chimaera":
        # TODO This section is temporary
        Grid = namedtuple(
            "Grid", "left right dx alpha active leftboundary rightboundary"
        )
        BoundaryOld = namedtuple("BoundaryOld", "type")
        active_cells = np.ones(num_cells)
        active_cells[75:100] = 0
        main = Grid(
            0,
            num_cells,
            dx_array,
            alpha_array,
            active_cells,
            BoundaryOld("periodic"),
            BoundaryOld("periodic"),
        )

        J = calc_jacobian_chimaera(num_cells, [main])

        solver_args = ([main], J)
    else:
        solver_args = (alpha_array, dx_array, J)

    solver_start_time = time.perf_counter()
    solver = solve_ivp(
        MODEL_LIST[args.model],
        (0, args.time),
        y,
        args=solver_args,
        method=args.solver,
        # jac=J,
    )
    solver_elapsed_time = time.perf_counter() - solver_start_time

    # Calculate the total energy for each timestep and the position of each cell
    if args.model == "diff_p_split" or args.model == "diff_chimaera":
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
