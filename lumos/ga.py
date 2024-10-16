import sys
import time
from datetime import datetime
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt

from lumos.aux.configs import Configs
from lumos.genetic_operators.select_methods import get_available_select_methods
from lumos.genetic_operators.crossover_methods import get_available_crossover_methods
from lumos.genetic_operators.mutation_methods import get_available_mutation_methods
from lumos.genetic_operators.init_population_methods import (
    get_available_init_population_methods,
)


plt.rcParams["font.size"] = 12


def is_odd(number):
    if number & 1:
        return True
    else:
        return False


class Ga:
    def __init__(self, config_file, f_obj, h_const=None, g_const=None):
        if config_file is None or f_obj is None:
            print('Os parâmetros "config_file" e "f_obj" não podem ser nulos.')
            raise RuntimeError(
                'Os parâmetros "config_file" e "f_obj" não podem ser nulos.'
            )

        # Carregamento do arquivo de configurações
        self.configs = Configs(config_file)

        # Requisitos de projeto fornecidos pelo usuário (alguns contendo valores padrão)
        self.f_obj = f_obj
        self.h_const = h_const
        self.g_const = g_const
        self.max_gen = None
        self.pop_len = None
        self.x_len = None
        self.select_method = None
        self.cross_method = None
        self.mut_method = None
        self.x_l = None
        self.x_u = None
        self.elitism_rate = None
        self.restrictions_weight = None
        self.gene_type = None
        self.bound_constraints_processing = None

        # Outros atributos
        self.start_time = time.time()
        self.pop = None
        self.f_obj_values = None
        self.f_obj_history = []
        self.best_x_history = []
        self.pop_history = []
        self.f_obj_calls = 0
        self.children_number = None
        self.children = None
        self.num_individuals_to_select = None
        self.num_individuals_to_mantain = None
        self.gen = 0

        self.check_input_messages = {
            "f_obj": "A função objetivo (f_obj) não foi informada.",
            "max_gen": "O número máximo de gerações (max_gen) não foi informado.",
            "pop_len": "O tamanho da população (pop_size) não foi informado.",
            "x_len": "A quantidade de genes do indivíduo (individual_size) não foi informada.",
            "select_method": "O método a ser empregado na seleção (select_type) não foi informado.",
            "cross_method": "O método a ser empregado na recombinação (cross_method) não foi informado.",
            "mut_method": "O método a ser empregado na mutação (mut_method) não foi informado.",
            "x_l": "Os limites inferiores para os genes dos indivíduos (solution_low_limits) não foram informados.",
            "x_u": "Os limites superiores para os genes dos indivíduos (solution_upper_limit) não foram informados.",
            "h_const": "O método h_const (que define as restrições de igualdade), deve obrigatóriamente retornar uma "
            "lista.",
            "g_const": "O método g_const (que define as restrições de desigualdade), deve obrigatóriamente retornar "
            "uma lista.",
            "gene_type": "O tipo de gene (gene_type) não foi informado.",
        }

        # Métodos disponíveis para implantação dos operadores genéticos
        self.available_select_methods = get_available_select_methods()
        self.available_crossover_methods = get_available_crossover_methods()
        self.available_mutation_methods = get_available_mutation_methods()
        self.available_init_population_methods = get_available_init_population_methods()

        # Definição do logger
        log_level = self.configs.get_config_else(None, "log_level").upper()
        if log_level is not None:
            logger.remove()
            logger.add(
                self.configs.get_config_else("output.log", "log_path"),
                format="{time} | {level} | {message}",
                level=log_level,
            )
            logger.add(
                sys.stdout, format="{time} | {level} | {message}", level=log_level
            )

        # Definição da semente empregada na geração de números aleatórios
        random_seed = self.configs.get_config_else(int(time.time()), "random_seed")
        logger.info(
            f"Semente utilizada na geração dos números aleatórios: {random_seed}."
        )
        self.rnd = np.random.default_rng(random_seed)

    def optimize(self):
        try:
            logger.info(
                f"Iniciando processo de otimização "
                f'({datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")}).'
            )
            self.config()
            self.init_population()
            self.run()
            results = self.show_results()
            logger.info("Processo de otimização concluído com sucesso.")
            return results
        except Exception as ex:
            logger.error(f"Ocorreu uma falha durante o processo de otimização ({ex}).")
            raise ex

    def config(self):
        self.check_input_params()
        self.compute_children_number()

    def check_input_params(self):
        logger.info("Iniciando checagem dos parâmetros fornecidos pelo usuário.")

        self.max_gen = self.check_param("max_gen")
        self.pop_len = self.check_param("pop_len")
        self.x_len = self.check_param("x_len")
        self.select_method = self.check_param("select_method")
        self.cross_method = self.check_param("cross_method")
        self.mut_method = self.check_param("mut_method")
        self.x_l = self.check_param("x_l")
        self.x_u = self.check_param("x_u")
        self.gene_type = self.check_param("gene_type")
        self.elitism_rate = self.configs.get_config_else(0.1, "elitism_rate")
        self.restrictions_weight = self.configs.get_config_else(
            1000, "restrictions_weight"
        )
        self.bound_constraints_processing = self.configs.get_config_else(
            "truncate", "bound_constraints_processing"
        )

        # Verificação dos limites superior e inferior para os genes dos indivíduos (limites laterais)
        if len(self.x_l) > self.x_len or len(self.x_u) > self.x_len:
            logger.error(
                "Verifique as dimensões de x_l e x_u. Elas devem ser iguais ao x_len."
            )

        logger.info("Checagem dos parâmetros concluída com sucesso.")
        logger.info("Parâmetros fornecidos pelo usuário:")
        logger.info(f"     - Número máximo de gerações: {self.max_gen}.")
        logger.info(f"     - Tamanho da população: {self.pop_len}.")
        logger.info(
            f"     - Comprimento do indivíduo (quantidade de genes): {self.x_len}."
        )
        logger.info(f"     - Método de seleção: {self.select_method}.")
        logger.info(f"     - Método de recombinação (crossover): {self.cross_method}.")
        logger.info(f"     - Método de mutação: {self.mut_method}.")

    def check_param(self, var_name):
        try:
            return self.configs.get_config(var_name)
        except RuntimeError:
            logger.error(self.check_input_messages[var_name])
            raise RuntimeError(self.check_input_messages[var_name])

    def compute_children_number(self):
        self.num_individuals_to_mantain = np.round(
            self.pop_len * self.elitism_rate
        ).astype(np.int64)
        if self.elitism_rate >= 1e-10:  # > 0
            if self.num_individuals_to_mantain == 0:
                self.num_individuals_to_mantain = 1
                logger.info(
                    "Assumindo que um único indivíduo (com o melhor fitness) será mantido a cada geração. "
                    "Para que não haja elitismo, defina elitism_rate = 0."
                )
            else:
                logger.info(
                    f"Assumindo que {self.num_individuals_to_mantain} indivíduos serão mantidos a cada geração"
                    f" (elitismo)."
                )
        else:
            logger.info("Não será considerado qualquer tipo de elitismo.")

        self.children_number = self.pop_len - self.num_individuals_to_mantain

        if is_odd(self.children_number):
            logger.info(
                "Garantindo que o número de filhos a serem produzidos seja par."
            )
            self.children_number -= 1
            self.num_individuals_to_mantain += 1

        self.num_individuals_to_select = self.children_number

        if self.num_individuals_to_select < 2:
            logger.error("Não é possível realizar o crossover com menos de dois pais.")
            raise RuntimeError(
                "Não é possível realizar o crossover com menos de dois pais."
            )

        logger.info(
            f"Será necessário selecionar {self.num_individuals_to_select} pais na etapa de seleção."
        )

    def init_population(self):
        if self.gene_type not in self.available_init_population_methods.keys():
            logger.error(f"Tipo de gene informado ({self.gene_type}) inválido.")
            raise RuntimeError(f"Tipo de gene informado ({self.gene_type}) inválido.")

        self.pop, self.f_obj_values = self.available_init_population_methods[
            self.gene_type
        ](self)
        self.show_best_x(0)

    def get_f_obj_values(self, pop):
        f_obj_values = np.empty(pop.shape[0])
        for i in range(pop.shape[0]):
            # Cálculo da penalidade referente às restrições de igualdade
            h_f_obj_impact = 0
            if self.h_const is not None:
                try:
                    h_f_obj_impact = np.sum(
                        [
                            np.abs(h_value) * self.restrictions_weight
                            for h_value in self.h_const(pop[i, :])
                        ]
                    )
                except TypeError as ex:
                    logger.error(
                        'O método "h_const" deve retornar, obrigatoriamente uma lista.'
                    )
                    raise RuntimeError(
                        f'O método "h_const" deve retornar, obrigatoriamente uma lista ({ex}).'
                    )

            # Cálculo da penalidade referente às restrições de desigualdade
            g_f_obj_impact = 0
            if self.g_const is not None:
                try:
                    g_f_obj_impact = np.sum(
                        [
                            np.max((g_value * self.restrictions_weight, 0))
                            for g_value in self.g_const(pop[i, :])
                        ]
                    )
                except TypeError as ex:
                    logger.error(
                        'O método "g_const" deve retornar, obrigatoriamente uma lista.'
                    )
                    raise RuntimeError(
                        f'O método "g_const" deve retornar, obrigatoriamente uma lista ({ex}).'
                    )

            # Cálculo da penalidade referente às restrições laterais
            l_f_obj_impact = 0
            if self.bound_constraints_processing != "truncate":
                l_f_obj_impact += np.sum(
                    [
                        np.max(((x_g - self.x_u[j]) * self.restrictions_weight, 0))
                        for j, x_g in enumerate(pop[i, :])
                    ]
                )
                l_f_obj_impact += np.sum(
                    [
                        np.max(((self.x_l[j] - x_g) * self.restrictions_weight, 0))
                        for j, x_g in enumerate(pop[i, :])
                    ]
                )

            # Computação da função objetivo
            f_obj_values[i] = (
                self.f_obj(pop[i, :]) - h_f_obj_impact - g_f_obj_impact - l_f_obj_impact
            )

            if np.isnan(f_obj_values[i]):
                logger.error(
                    f"Foi produzido um NaN durante a computação da função objetivo. Indivíduo: {pop[i, :]}"
                )
                logger.error("Atribuindo ao indivíduo em questão fitness igual a -inf.")
                f_obj_values[i] = -np.inf

            self.f_obj_calls += 1

        return f_obj_values

    def run(self):
        while True:
            logger.debug(f"Iniciando processamento da geração {self.gen + 1}.")
            select_individuals = self.select()
            children = self.crossover(select_individuals)
            mutate_children = self.mutation(children)
            self.build_new_pop(mutate_children)
            self.show_best_x(self.gen)
            if self.stop_now(self.gen):
                break

            logger.debug(
                f"Processamento da geração {self.gen + 1} concluído com sucesso."
            )
            self.gen += 1

        logger.info("Processo de iteração encerrado.")

    def select(self):
        if self.select_method not in self.available_select_methods.keys():
            logger.error(
                f"Método de seleção informado ({self.select_method}) não disponível."
            )
            raise RuntimeError(
                f"Método de seleção informado ({self.select_method}) não disponível."
            )

        try:
            return self.available_select_methods[self.select_method](self)
        except Exception as ex:
            logger.error(f"Falha na execução da seleção ({ex}).")
            raise ex

    def crossover(self, select_individuals):
        if self.cross_method not in self.available_crossover_methods.keys():
            logger.error(
                f"Método de recombinação informado ({self.cross_method}) não disponível."
            )
            raise RuntimeError(
                f"Método de recombinação informado ({self.cross_method}) não disponível."
            )

        try:
            return self.available_crossover_methods[self.cross_method](
                self, select_individuals
            )
        except Exception as ex:
            logger.error(f"Falha na execução da recombinação ({ex}).")
            raise ex

    def mutation(self, children):
        if self.mut_method not in self.available_mutation_methods.keys():
            logger.error(
                f"Método de mutação informado ({self.mut_method}) não disponível."
            )
            raise RuntimeError(
                f"Método de mutação informado ({self.mut_method}) não disponível."
            )

        try:
            return self.available_mutation_methods[self.mut_method](self, children)
        except Exception as ex:
            logger.error(f"Falha na execução da mutação ({ex}).")
            raise ex

    def build_new_pop(self, mutate_children):
        logger.debug("Iniciando construção da nova população.")
        f_obj_values_sort_index = np.argsort(-self.f_obj_values)[
            : self.num_individuals_to_mantain
        ]
        individuals_to_mantain = self.pop[f_obj_values_sort_index, :]

        self.pop = np.concatenate((individuals_to_mantain, mutate_children))
        self.bound_constraint_processing()
        self.f_obj_values = self.get_f_obj_values(self.pop)

        logger.debug("Nova população construída com sucesso.")

    def bound_constraint_processing(self):
        if self.bound_constraints_processing == "truncate":
            logger.debug(
                "Realizando o truncameno dos genes dos indivíduos da população."
            )
            for i in range(self.pop.shape[0]):
                for j in range(self.pop.shape[1]):
                    self.pop[i, j] = np.min((self.pop[i, j], self.x_u[j]))
                    self.pop[i, j] = np.max((self.pop[i, j], self.x_l[j]))

    def show_best_x(self, gen):
        best_fitness = np.max(self.f_obj_values)
        best_x = self.pop[np.argmax(self.f_obj_values), :]
        self.f_obj_history.append(best_fitness)
        self.best_x_history.append(best_x)
        self.pop_history.append(self.pop)
        logger.info(
            f"Geração: {gen} | Melhor fitness: {best_fitness} | Indivíduo: {best_x}"
        )

    def stop_now(self, gen):
        max_exec_time_seconds = self.configs.get_config_else(
            None, "max_exec_time_seconds"
        )
        min_f_obj_value_diff = self.configs.get_config_else(
            None, "min_f_obj_value_diff"
        )
        generations_to_check_f_obj_diff = self.configs.get_config_else(
            None, "generations_to_check_f_obj_diff"
        )
        logger.debug("Iniciando verificação dos critérios de parada.")

        if (
            max_exec_time_seconds is not None
            and time.time() - self.start_time > max_exec_time_seconds
        ):
            logger.info(f"Tempo máximo alcançado ({max_exec_time_seconds} segundos).")
            return True

        if (
            min_f_obj_value_diff is not None
            and gen > generations_to_check_f_obj_diff
            and np.all(
                np.abs(np.diff(self.f_obj_history[-generations_to_check_f_obj_diff:]))
                < min_f_obj_value_diff
            )
        ):
            logger.info("Não verificou-se alteração no valor da função objetivo.")
            return True

        if gen >= self.max_gen:
            logger.info("Alcançado o número máximo de iterações.")
            return True

        logger.debug("Verificação dos critérios de parada concluída com sucesso.")
        return False

    def show_results(self):
        exec_time = time.time() - self.start_time
        best_x = self.best_x_history[-1]
        best_f = self.f_obj_history[-1]
        logger.info(f"Melhor fitness obtido: {best_f}.")
        logger.info(f"Melhor solução obtida: {best_x}.")

        results = {
            "best_x": best_x,
            "best_f": best_f,
            "max_gen": self.gen,
            "f_calls": self.f_obj_calls,
            "exec_time": exec_time,
        }

        if self.h_const is not None:
            logger.info(
                f"Valores das restrições de igualdade (= 0): {[float(h) for h in self.h_const(best_x)]}."
            )
            results.update({"h_values": [float(h) for h in self.h_const(best_x)]})
        if self.g_const is not None:
            logger.info(
                f"Valores das restrições de desigualdade (< 0): {[float(g) for g in self.g_const(best_x)]}."
            )
            results.update({"g_values": [float(g) for g in self.g_const(best_x)]})

        logger.info(
            f"Tempo despendido na execução: {exec_time:.2f} s "
            f"({exec_time / 60:.2f} minutos)."
        )
        logger.info(f"Número de gerações: {self.gen}.")
        logger.info(f"Número de avaliações da função objetivo: {self.f_obj_calls}.")

        plt.figure()
        plt.plot(self.f_obj_history, linewidth=1.5)
        plt.plot(self.f_obj_history, ".", markersize=5, color="tab:blue")
        plt.title("Evolução do fitness ao longo das gerações")
        plt.xlabel("Geração")
        plt.ylabel("Fitness")
        plt.grid(linestyle="--")
        plt.tight_layout()
        plt.savefig("fitness_history.eps")

        # Representação gráfica dos resultados
        if self.configs.get_config_else(None, "plot_f_obj_history"):
            plt.show()

        return results

    def maxima(self):
        self.optimize()
