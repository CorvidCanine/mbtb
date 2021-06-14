import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import animation

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


if __name__ == "__main__":

    y = np.zeros(100)
    y[50] = 1000
    alpha = 1
    dx = 1

    solver = solve_ivp(diffusion_periodic, (0, 50000), y, args=(alpha, dx), method="RK45")
    print(solver)

    energy_array = np.sum(solver.y, axis=0)
    
    print("Start energy", energy_array[0], "end energy", energy_array[-1])
    print("Energy diff", energy_array[-1] - energy_array[0])

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

    # # Set up formatting for the movie files
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=60, bitrate=1800)

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

    ani.save("out.gif", writer='imagemagick')

    plt.show()

    plt.plot(solver.y[:,-1])
    plt.plot(solver.y[:,0])

    plt.xlabel("Position")
    plt.ylabel("Value (\"Temperture\") ")

    plt.show()