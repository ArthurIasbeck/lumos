from resources.config import logger
import numpy as np


def get_available_select_methods():
    return {"roulette": roulette}


def roulette(ga_data):
    logger.debug("Iniciando etapa de seleção (aplicação do Método da Roleta)")

    # Tratamento dos fitness negativos
    if np.min(ga_data.f_obj_values) < 0:
        f_obj_values = ga_data.f_obj_values + 1.1 * np.abs(np.min(ga_data.f_obj_values))
    else:
        f_obj_values = ga_data.f_obj_values

    f_obj_values[f_obj_values == 0] = (
        np.max(f_obj_values) * 0.05 if np.max(f_obj_values) != 0 else 0.1
    )

    # Normalização dos fitness
    f_obj_sum = np.sum(f_obj_values)
    f_obj_values = f_obj_values / f_obj_sum

    # Construção da roleta
    roulette_limits = np.empty(ga_data.pop_len)

    for i, f_obj in enumerate(f_obj_values):
        if i == 0:
            roulette_limits[i] = f_obj_values[i]
        else:
            roulette_limits[i] = roulette_limits[i - 1] + f_obj_values[i]

    # Sorteio dos indivíduos pais para a etapa de crossover
    select_individuals = np.empty((ga_data.num_individuals_to_select, ga_data.x_len))
    for i in range(ga_data.num_individuals_to_select):
        random_number = ga_data.rnd.random()
        try:
            select_index = np.nonzero(roulette_limits > random_number)[0][0]
        except:
            print("ERROR")
        select_individuals[i, :] = ga_data.pop[select_index, :]

    logger.debug("Seleção concluída com sucesso.")
    return select_individuals
