import toml


class Configs:
    def __init__(self, toml_file_name):

        try:
            with open(toml_file_name, "r") as config_file:
                self.configs_from_toml = toml.load(config_file)
        except Exception as ex:
            raise RuntimeError(
                f"Falha no carregamento do arquivo de configurações ({ex})."
            )

    def config_exists(self, var_name, group_name=None):
        try:
            self.get_config(var_name, group_name)
            return True
        except RuntimeError:
            return False

    def get_config_else(self, else_value, var_name, group_name=None):
        try:
            return self.get_config(var_name, group_name)
        except RuntimeError:
            return else_value

    def get_config(self, var_name, group_name=None):
        if group_name is None:
            if var_name in self.configs_from_toml.keys():
                return self.configs_from_toml[var_name]
            else:
                raise RuntimeError(f'Variável "{var_name}" não encontrada.')

        else:
            if (
                group_name in self.configs_from_toml.keys()
                and var_name in self.configs_from_toml[group_name].keys()
            ):
                return self.configs_from_toml[group_name][var_name]
            else:
                raise RuntimeError(
                    f'Variável "{var_name}" (pertencente ao grupo "{group_name}") não encontrada. '
                    f"Verifique os nomes do grupo e da variável."
                )
