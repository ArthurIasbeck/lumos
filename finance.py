import numpy as np

from resources.config import init
from src.ga import Ga
import matplotlib.pyplot as plt


def f_obj(x):
    w_1 = x[0]
    w_2 = x[1]
    return -(0.25 * w_1**2 + 0.1 * w_2**2 + 0.3 * w_1 * w_2)


def h_const(x):
    w_1 = x[0]
    w_2 = x[1]
    return [w_1 + w_2 - 1]


def g_const(x):
    w_1 = x[0]
    g_1 = -w_1
    g_2 = w_1 - 1
    return [g_1, g_2]


def plot_f_obj():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    X, Y = np.meshgrid(x, y)
    Z = 0.25 * X**2 + 0.1 * Y**2 + 0.3 * X * Y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="inferno")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    levels = np.linspace(np.min(Z), np.max(Z), 10)
    fig, ax = plt.subplots()
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap="inferno")
    fig.colorbar(contourf)

    plt.show()


def main():
    plot_f_obj()
    ga = Ga(
        config_file="configs_finance.toml",
        f_obj=f_obj,
        h_const=h_const,
        g_const=g_const,
    )
    result = ga.optimize()
    print(result)


if __name__ == "__main__":
    init()
    main()
