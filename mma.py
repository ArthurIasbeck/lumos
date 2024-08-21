import numpy as np
from loguru import logger
from resources.config import init
from src.ga import Ga

# Variávei de otimização
# N = x[0]
# I_max = x[1]
# c = x[2]
# h_w = x[3]

D_sh = 33 / 1000
c_g = 0.5 / 1000
F_max = 120
mu_0 = 4 * np.pi * 1e-7
epsilon = 0.8
B_max = 1.3
alpha = np.deg2rad(22.5)
J_cu = 6e6

A_a = (F_max * mu_0) / (B_max**2 * epsilon * np.cos(alpha))
D_i = 8 * (D_sh + 2 * c_g) / (8 - np.pi)


def f_obj(x):
    # Variáveis de otimização
    c = x[2]
    h_w = x[3]

    # Computação de parâmetros
    D_st = D_i + 2 * (c + 1.15 * h_w)

    # Computação da função objetivo
    V = (np.pi * D_st**2 * A_a) / (4 * c)
    return -V


def h_const(x):
    N = x[0]
    I_max = x[1]

    h_1 = N * I_max - ((2 * B_max * c_g) / mu_0)
    h_2 = F_max - (epsilon * mu_0 * N**2 * I_max**2 * A_a * np.cos(alpha)) / (
        4 * c_g**2
    )

    return [h_1, h_2]


def g_const(x):
    N = x[0]
    I_max = x[1]
    c = x[2]
    h_w = x[3]

    phi = np.sqrt((4 * I_max) / (np.pi * J_cu))
    h = 1.5 * h_w

    g_1 = (N * np.pi * phi**2) / 8 - 0.8 * (
        alpha / 2 * (D_i / 2 + h) ** 2
        - alpha / 2 * (D_i / 2) ** 2
        - c * h / 2
        - alpha / 2 * h_w**2
    )
    g_2 = (
        (N * np.pi * phi**2) / (8 * h_w)
        - (2 * (D_i / 2 + 0.15 * h_w)) * np.sin(alpha / 2)
        - c / 2
    )
    return [g_1, g_2]


def main():
    ga = Ga(
        config_file="configs_mma.toml", f_obj=f_obj, h_const=h_const, g_const=g_const
    )
    result = ga.optimize()

    # Apresentação do resultado obtido
    best_x = result["best_x"]
    N = best_x[0]
    I_max = best_x[1]
    c = best_x[2]
    h_w = best_x[3]
    logger.info(f"N = {np.int64(np.round(N))}")
    logger.info(f"I_max = {I_max:.3f} A")
    logger.info(f"c = {c * 1000:.3f} mm")
    logger.info(f"h_w = {h_w * 1000:.3f} mm")

    # Comparação com a literatura
    J_ga = -f_obj([308.1, 4.95, 9.6 / 1000, 37.8 / 1000])
    J_ps = -f_obj([308.64, 4.94, 4.01 / 1000, 36.48 / 1000])
    J_lumos = -f_obj([N, I_max, c, h_w])
    logger.info(
        f"GA: {J_ga * 1e6:.4f} cm³ | PS: {J_ps * 1e6:.4f} cm³ | Lumos: {J_lumos * 1e6:.4f} cm³"
    )


if __name__ == "__main__":
    init()
    main()
