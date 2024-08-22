# _Lumos_

O *Lumos* é um pacote Python aberto que possibilita a resolução de problemas de otimização por meio do emprego de Algoritmos Genéticos (AGs) e, mais especificamente, da seleção via roleta, da recombinação aritmética e da mutação não uniforme gaussiana. Diferentemente de outros pacotes que possibilitam a implementação de AGs, o *Lumos* se baseia na utilização de operadores genéticos que operam diretamente sobre variáveis de otimização reais (de ponto flutuante), o que possibilita que problema de otimização baseados em variáveis desse tipo sejam resolvidos de forma mais eficiente. 

# Instalando o Lumos

Para utilizar *Lumos*, basta adicionar a linha abaixo ao seu arquivo `requirements.txt` 

```latex
lumos @ git+https://github.com/ArthurIasbeck/lumos@main
```

ou, se preferir, instalar o pacote diretamente, utilizando o comando 

```latex
pip install git+https://github.com/ArthurIasbeck/lumos@main
```

# Utilizando o Lumos

A utilização do *Lumos* depende da construção de um arquivo TOML, em que são definidos os parâmetros nos quais a solução do problema de otimização se baseia. Além disso, é necessário construir *script* no qual o *Lumos* será instanciado e serão definidas a função objetivo (fitness) a ser maximizada e as restrições associadas ao problema de otimização a ser resolvido. O exemplo abaixo mostra como o *Lumos* foi empregado na resolução de um problema de otimização relacionado a finanças. A pergunta a ser respondida é: se tivermos 1 dólar e nos envolvermos em dois investimentos diferentes, nos quais seu retorno é modelado como uma distribuição gaussiana bivariada. Quanto devemos investir em cada um para minimizar a variância geral no retorno? O problema em questão pode ser formulado conforme segue:

$$ \mathrm{f}(\mathrm{w}_1,\mathrm{w}_2)=0.25 \mathrm{w}_1^2+0.1\mathrm{w}_2^2+0.3\mathrm{w}_1\mathrm{w}_2 $$

$$ \mathrm{w}_1 + \mathrm{w}_2 = 1 $$

$$ \mathrm{w}_1 \geq 0 $$

$$ \mathrm{w}_1 \leq 1$$

Primeiramente montamos o arquivo de configurações TOML, que nesse caso receberá o nome de `configs_finance.toml`:

```python
# Parâmetros base da otimização
max_gen = 35  # Número máximo de gerações
pop_len = 100  # Tamanho da população
x_len = 2  # Número de genes em cada indivíduo (x)
elitism_rate = 0.2  # Percentual de indivíduos na população atual que serão mantidos na população seguinte
mutation_rate = 0.1  # Probabilidade de um determinado indivíduo sofrer mutação
#random_seed = 0  # Semente para geração dos números aleatórios. Caso não seja definida, assumirá o valor de time.time()

# Métodos a serem empregados nas operações executadas pelo AG (operadores genéticos)
select_method = 'roulette'  # Método de seleção a ser empregado
cross_method = 'arithmetic_recombination'  # Método empregado na recombinação (crossover)
mut_method = 'nonuniform_gaussian'  # Método empregado na mutação
gene_type = 'real'  # Tipo numérico dos genes do indivíduo (dita como a primeira população será inicializada)

# Parâmetros associados às restrições
x_l = [0.0, 0.0]  # Limites inferiores para cada um dos genes do indivíduo
x_u = [1.0, 1.0]  # Limites superiores para cada um dos genes do indivíduo
restrictions_weight = 1000  # Peso que determina o impacto do desrespeito das restrições na função objetivo

# Parâmetros associados às condições de parada
#max_exec_time_seconds = 3600  # Duração máxima da execução

## Se a variação da função objetivo (levando-se em conta o melhor indivíduo) for menor do que "min_f_obj_value_diff"
## por "generations_to_check_f_obj_diff" gerações, a otimização será encerrada
#min_f_obj_value_diff = 1e-6
#generations_to_check_f_obj_diff = 15

# Parâmetros associados à apresentação dos resultados
plot_f_obj_history = true  # Define se a evolução da função objetivo será representada graficamente
log_level = 'debug'  # Nível dos logs apresentados ("debug", "info", "error" ou None)
log_path = 'finance.log'  # Nome do arquivo onde os logs serão armazenados

# Parâmetros associados à recombinação. Os parâmetros a serem definidos podem variar de acordo com o método escolhido
[crossover]
alpha = 0.3  # Fator que define o peso que cada pai terá na computação de seus filhos (sugere-se 0 < alpha < 0.5)

# Parâmetros associados à mutação. Os parâmetros a serem definidos podem variar de acordo com o método escolhido
[mutation]
reduce_mut_factor = 6  # Quanto maior for o "reduce_mut_factor", mais sutil será a alteração provocada pela mutação
```

Por fim, define-se o script principal: 

```python
import lumos

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

def main():
    ga = lumos.Ga(
        config_file="configs_finance.toml",
        f_obj=f_obj,
        h_const=h_const,
        g_const=g_const,
    )
    result = ga.optimize()
    print(result)

if __name__ == "__main__":
    main()

```

O exemplo em questão pode ser verificado no diretório `examples`. 

# Representação dos resultados

O método `Ga.optimize()` retorna um objeto que contém várias informações referentes à solução encontrada. Além disso, se a variável `log_level` do arquivo de configurações assumir um valor diferente de `None` a evolução do processo de otimização é apresentada na tela. Caso se pretenda representar graficamente a evolução do valor da função objetivo ao longo das gerações, basta que se atribua à variável `plot_f_obj_history` o valor `True`. 

Abaixo se encontra um exemplo de um *log* bem-sucedido assim como de um gráfico que representa a evolução do valor da função objetivo. 

 

```python
2024-08-21T17:54:02.084218-0300 | INFO | Semente utilizada na geração dos números aleatórios: 1724273642.
2024-08-21T17:54:02.089750-0300 | INFO | Iniciando processo de otimização (2024-08-21 17:54:02).
2024-08-21T17:54:02.089996-0300 | INFO | Iniciando checagem dos parâmetros fornecidos pelo usuário.
2024-08-21T17:54:02.090087-0300 | INFO | Checagem dos parâmetros concluída com sucesso.
2024-08-21T17:54:02.090146-0300 | INFO | Parâmetros fornecidos pelo usuário:
2024-08-21T17:54:02.090197-0300 | INFO |      - Número máximo de gerações: 35.
2024-08-21T17:54:02.090247-0300 | INFO |      - Tamanho da população: 100.
2024-08-21T17:54:02.090297-0300 | INFO |      - Comprimento do indivíduo (quantidade de genes): 2.
2024-08-21T17:54:02.090345-0300 | INFO |      - Método de seleção: roulette.
2024-08-21T17:54:02.090393-0300 | INFO |      - Método de recombinação (crossover): arithmetic_recombination.
2024-08-21T17:54:02.090440-0300 | INFO |      - Método de mutação: nonuniform_gaussian.
2024-08-21T17:54:02.090573-0300 | INFO | Assumindo que 20 indivíduos serão mantidos a cada geração (elitismo).
2024-08-21T17:54:02.090638-0300 | INFO | Será necessário selecionar 80 pais na etapa de seleção.
2024-08-21T17:54:02.090692-0300 | DEBUG | Definição dos indivíduos da população inicial (em que os genes são números reais).
2024-08-21T17:54:02.090860-0300 | DEBUG | Inicialização do vetor que armazena os fitness dos indivíduos da população.
2024-08-21T17:54:02.092835-0300 | INFO | Geração: 0 | Melhor fitness: -1.1341122824823304 | Indivíduo: [0.69104912 0.30989175]
2024-08-21T17:54:02.092953-0300 | DEBUG | Iniciando processamento da geração 1.
2024-08-21T17:54:02.092999-0300 | DEBUG | Iniciando etapa de seleção (aplicação do Método da Roleta)
2024-08-21T17:54:02.093315-0300 | DEBUG | Seleção concluída com sucesso.
2024-08-21T17:54:02.093370-0300 | DEBUG | Iniciando etapa de recombinação (método da recombinação aritmética).
2024-08-21T17:54:02.093470-0300 | DEBUG | Recombinação concluída com sucesso.
2024-08-21T17:54:02.093508-0300 | DEBUG | Iniciando processo de mutação.
2024-08-21T17:54:02.093592-0300 | DEBUG | Mutação concluída com sucesso (4 mutados).
2024-08-21T17:54:02.093632-0300 | DEBUG | Iniciando construção da nova população.
2024-08-21T17:54:02.093703-0300 | DEBUG | Realizando o truncameno dos genes dos indivíduos da população.
2024-08-21T17:54:02.096448-0300 | DEBUG | Nova população construída com sucesso.
2024-08-21T17:54:02.096605-0300 | INFO | Geração: 0 | Melhor fitness: -1.1341122824823304 | Indivíduo: [0.69104912 0.30989175]
2024-08-21T17:54:02.096662-0300 | DEBUG | Iniciando verificação dos critérios de parada.
2024-08-21T17:54:02.096700-0300 | DEBUG | Verificação dos critérios de parada concluída com sucesso.
2024-08-21T17:54:02.096750-0300 | DEBUG | Processamento da geração 1 concluído com sucesso.
2024-08-21T17:54:02.096800-0300 | DEBUG | Iniciando processamento da geração 2.
...
2024-08-21T17:54:02.241894-0300 | INFO | Geração: 35 | Melhor fitness: -0.16556979093177612 | Indivíduo: [0.5122836  0.48771518]
2024-08-21T17:54:02.241948-0300 | DEBUG | Iniciando verificação dos critérios de parada.
2024-08-21T17:54:02.241993-0300 | INFO | Alcançado o número máximo de iterações.
2024-08-21T17:54:02.242027-0300 | INFO | Processo de iteração encerrado.
2024-08-21T17:54:02.242063-0300 | INFO | Melhor fitness obtido: -0.16556979093177612.
2024-08-21T17:54:02.242154-0300 | INFO | Melhor solução obtida: [0.5122836  0.48771518].
2024-08-21T17:54:02.242206-0300 | INFO | Valores das restrições de igualdade (= 0): [-1.2200123395977869e-06].
2024-08-21T17:54:02.242245-0300 | INFO | Valores das restrições de desigualdade (< 0): [-0.5122836049744728, -0.4877163950255272].
2024-08-21T17:54:02.242280-0300 | INFO | Tempo despendido na execução: 0.16 s (0.00 minutos).
2024-08-21T17:54:02.242312-0300 | INFO | Número de gerações: 35.
2024-08-21T17:54:02.242343-0300 | INFO | Número de avaliações da função objetivo: 3700.
2024-08-21T17:54:03.711927-0300 | INFO | Processo de otimização concluído com sucesso.
```

![Figure_1](https://github.com/user-attachments/assets/56d9c63d-4226-44b8-987c-59b8f5abb5e5)

# Licença

Este projeto está licenciado sob a [Licença Pública Geral GNU v3.0](https://choosealicense.com/licenses/gpl-3.0/).

A Licença Pública Geral GNU concede a você os seguintes direitos em relação a este software:

- **Liberdade de executar o programa como desejar,** para qualquer propósito.
- **Liberdade de estudar como o programa funciona,** e alterá-lo para que ele faça o que você quiser.
- **Liberdade de redistribuir cópias** para que você possa ajudar outros.
- **Liberdade de distribuir cópias de suas versões modificadas** para outros. Ao fazer isso, você pode dar a outras pessoas a chance de se beneficiarem das mudanças que você fez.

Para mais informações, veja o arquivo [LICENSE](https://www.notion.so/arthur-iasbeck/LICENSE) neste repositório.

# Melhorias futuras

- Traduzir o pacote para o inglês.
- Permitir que o usuário carregue os parâmetros de configuração a partir de um arquivo ou de um dicionário. Isso facilitaria a execução de análise de sensibilidade.
