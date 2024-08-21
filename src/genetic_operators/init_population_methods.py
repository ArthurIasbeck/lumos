from resources.config import logger
import numpy as np


def get_available_init_population_methods():
    return {"real": real}


def real(ga_data):
    logger.debug(
        "Definição dos indivíduos da população inicial (em que os genes são números reais)."
    )
    pop = np.empty((ga_data.pop_len, ga_data.x_len))
    for i in range(ga_data.pop_len):
        for j in range(ga_data.x_len):
            pop[i, j] = ga_data.x_l[j] + ga_data.rnd.random() * (
                ga_data.x_u[j] - ga_data.x_l[j]
            )

    logger.debug(
        "Inicialização do vetor que armazena os fitness dos indivíduos da população."
    )
    f_obj_values = ga_data.get_f_obj_values(pop)
    return pop, f_obj_values
