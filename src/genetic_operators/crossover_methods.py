from resources.config import logger
import numpy as np


def get_available_crossover_methods():
    return {"arithmetic_recombination": arithmetic_recombination}


def arithmetic_recombination(ga_data, select_individuals):
    logger.debug("Iniciando etapa de recombinação (método da recombinação aritmética).")
    alpha = ga_data.configs.get_config("alpha", "crossover")
    children = np.empty((ga_data.children_number, ga_data.x_len))
    for i in range(0, ga_data.children_number, 2):
        for j in range(ga_data.x_len):
            children[i, j] = (
                alpha * select_individuals[i, j]
                + (1 - alpha) * select_individuals[i + 1, j]
            )
            children[i + 1, j] = (1 - alpha) * select_individuals[
                i, j
            ] + alpha * select_individuals[i + 1, j]

    logger.debug("Recombinação concluída com sucesso.")
    return children
