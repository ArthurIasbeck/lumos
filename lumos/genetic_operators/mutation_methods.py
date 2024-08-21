from loguru import logger
import numpy as np


def get_available_mutation_methods():
    return {"nonuniform_gaussian": nonuniform_gaussian}


def nonuniform_gaussian(ga_data, children):
    logger.debug("Iniciando processo de mutação.")
    mutation_rate = ga_data.configs.get_config("mutation_rate")
    reduce_mut_factor = ga_data.configs.get_config("reduce_mut_factor", "mutation")
    mutate_children = np.array(children)
    count_mutations = 0
    for i in range(ga_data.children_number):
        if ga_data.rnd.random() <= mutation_rate:
            count_mutations += 1
            for j in range(ga_data.x_len):
                std = (ga_data.x_u[j] - ga_data.x_l[j]) / reduce_mut_factor
                mutate_children[i, j] += ga_data.rnd.normal(0, std)

    logger.debug(f"Mutação concluída com sucesso ({count_mutations} mutados).")
    return mutate_children
