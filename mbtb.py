import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
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
    y1[0] = alpha * (y[-1] - 2 * y[0] + y[1]) / 1 ** 2
    y1[-1] = alpha * (y[-2] - 2 * y[-1] + y[0]) / 0.5 ** 2

    return y1

if __name__ == "__main__":

    if False:
        # Non-split scenario

        y = np.zeros(200)
        y[25] = 1000
        alpha = 1
        dx = 0.5

        solver_start_time = time.perf_counter()
        solver = solve_ivp(diffusion_periodic, (0, 2000), y, args=(alpha, dx), method="RK45")
        solver_elapsed_time = time.perf_counter() - solver_start_time

        energy_array = np.sum(solver.y, axis=0) * 0.5
    else:
        # Split scenario

        y = np.zeros(150)
        y[25] = 1000
        alpha = 1
        Submesh = namedtuple("Submesh", "start end dx")
        # rough description of the mesh made of submeshes with different cell widths
        split_mesh = [Submesh(1,50,1), Submesh(50,149,0.5)]

        solver_start_time = time.perf_counter()
        solver = solve_ivp(diffusion_periodic_split, (0, 2000), y, args=(alpha, split_mesh), method="RK45")
        solver_elapsed_time = time.perf_counter() - solver_start_time
        
        # Calculate the total energy for each timestep
        energy_array = np.zeros(len(solver.t))
        for m in split_mesh:
            energy_array[:] += np.sum(solver.y[m.start:m.end,:], axis=0) * m.dx
        # The boundary cells are currently not included in the mesh description
        energy_array[:] += solver.y[0,:] * 1
        energy_array[:] += solver.y[-1,:] * 0.5

        
    print(solver)
    print(f"Elapsed time for solver was {solver_elapsed_time} seconds")
    print("Start energy", energy_array[0], "end energy", energy_array[-1])
    print("Energy diff", energy_array[-1] - energy_array[0])

    # Various plotting stuff this point onward

    if True:
        plt.imshow(
            solver.y,
            cmap="inferno",
            aspect="auto",
            interpolation="none"
        )

        plt.colorbar()

        plt.xlabel("Time")
        plt.ylabel("Position")

        plt.show()

    if True:
        print(len(solver.t))
        fig, ax = plt.subplots()
        frames = []
        for step in range(0,len(solver.t),100):
            frames.append([ax.imshow(
                    np.expand_dims(solver.y[:,step], axis=0),
                    cmap="inferno",
                    animated=True,
                    aspect="auto"
                )]
            )
        ani = animation.ArtistAnimation(fig, frames, interval=60, blit=True, repeat_delay=2000)

        # ani.save("out.gif", writer='imagemagick')

        plt.show()

    plt.plot(solver.y[:,-1])
    plt.plot(solver.y[:,0])

    plt.xlabel("Cell") # Cell index arn't corrected to positions yet
    plt.ylabel("Value (\"Temperture\")")

    plt.show()