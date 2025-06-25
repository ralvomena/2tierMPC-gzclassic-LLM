# 2tierMPC-gzclassim-LLM

Com certeza\! Usei o seu arquivo `README.md` mais recente como base, mantive as instruções originais e integrei todas as seções referentes à configuração e execução com o Ollama e o LLM, como havíamos discutido.

Também preservei a instrução de instalar o **Gazebo 11 (Classic)**, conforme sua solicitação, em vez da versão Citadel mencionada no `README.md` que você anexou, garantindo consistência com a dependência original do projeto.

O resultado é um documento único e coeso, pronto para ser usado no seu repositório.

-----

# 2tierMPC-LLM: Instruções de Execução e Simulação

## Visão Geral

Este documento descreve os passos para configurar o ambiente e executar os cenários de simulação do projeto 2tierMPC. Ele cobre a instalação das ferramentas base (ROS 2, Gazebo), a compilação dos pacotes do projeto e as instruções específicas para executar os diferentes cenários, com destaque para o **Cenário 2**, que integra um Modelo de Linguagem de Grande Escala (LLM) para otimização da linha de produção.

**Nota Importante:** Atualmente, a integração com o LLM está implementada exclusivamente para o **Cenário 2**. Os demais cenários do projeto (1, 3 e 4) permanecem operacionais e podem ser executados normalmente, porém sem as funcionalidades de otimização via LLM.

## 1\. Instalação do Ambiente e Dependências

As simulações foram validadas com o sistema operacional, ferramentas e pacotes indicados a seguir:

  * Ubuntu 20.04 LTS arm64
  * ROS 2 Foxy Fitzroy
  * Gazebo 11 (Classic)
  * Python 3.8.10
  * CasADi v3.6.7

### 1.1. Ferramentas base (ROS/Gazebo)

A instalação das ferramentas pode ser feita em dois computadores/máquinas virtuais, uma representando o lado da borda e a outra o lado local, ou apenas em um computador/máquina virtual.

1.  **Instalar o ROS 2 Foxy:** Siga as instruções detalhadas neste [link](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html) (incluindo o `ros-dev-tools`).

2.  **Configurar o ambiente ROS 2:** Para que os comandos do ROS 2 se tornem acessíveis, execute no terminal (apenas uma vez):

    ```bash
    echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
    ```

    Veja mais detalhes neste [link](https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html).

3.  **Instalar o CasADi para Python:** Siga as instruções de download e instalação neste [link](https://web.casadi.org/get/).

4.  **Instalar o Gazebo 11 (apenas no lado local):** Siga as instruções de instalação neste [link](https://www.google.com/search?q=http://gazebosim.org/tutorials%3Ftut%3Dinstall_ubuntu%26cat%3Dinstall).

5.  **Instalar os pacotes de integração ROS/Gazebo (apenas no lado local):** Execute os comandos:

    ```bash
    sudo apt-get install ros-foxy-gazebo-ros-pkgs
    ```

### 1.2. Ferramentas para integração com LLM (apenas Cenário 2)

Para executar a funcionalidade de otimização com o LLM, as seguintes ferramentas são necessárias.

1.  **Instalar o Ollama:** Ollama é a ferramenta utilizada para executar modelos de linguagem de grande escala localmente. Siga as instruções de instalação para o seu sistema operacional no site oficial: [ollama.com](https://ollama.com).

2.  **Baixar o Modelo LLM:** Após instalar o Ollama, baixe o modelo que será utilizado pelo cenário. Abra um terminal e execute:

    ```bash
    ollama pull llama3.2
    ```

3.  **Instalar Módulo Python `requests`:** Para que o código da simulação possa se comunicar com a API do Ollama, a biblioteca `requests` é necessária.

    ```bash
    pip3 install requests
    ```

## 2\. Preparação dos pacotes do projeto

### 2.1. Pacotes da Borda (`edge`)

1.  Abra um terminal no diretório `/edge` disponibilizado e execute o comando:
    ```bash
    colcon build
    ```
2.  Caso queira modificar os códigos dos pacotes sem a necessidade de executar novamente o `colcon`, utilize a opção `--symlink-install`:
    ```bash
    colcon build --symlink-install
    ```
3.  Para que os pacotes construídos se tornem acessíveis, execute o comando abaixo, colocando o caminho até o diretório `/edge`:
    ```bash
    echo "source /caminho_ate_o_diretorio/edge/install/setup.bash" >> ~/.bashrc
    ```

### 2.2. Pacotes do lado local (`local`)

Nas simulações realizadas, os pacotes da borda e do lado local foram executados em dois computadores diferentes. Para a preparação dos pacotes no lado local, a mesma sequência de comandos deve ser executada a partir do diretório `/local`:

```bash
colcon build # adicione o --sylink-install para alteração do código sem a necessidade de reexecutar o colcon 
echo "source /caminho_ate_o_diretorio/local/install/setup.bash" >> ~/.bashrc
```

**Atenção:** Lembre-se de substituir `/caminho_ate_o_diretorio/` pelo caminho absoluto correto em sua máquina e de abrir um novo terminal para que as alterações tenham efeito.

## 3\. Executando a Simulação

### 3.1. Configuração do Cenário 2 (Com LLM)

1.  **Iniciar o Servidor Ollama:** Conforme a seção 1.2, garanta que o servidor Ollama esteja rodando e acessível na rede.

2.  **Alterar a Classe do Supervisor:** Abra o arquivo `supervisor.py` no pacote `edge_tier_pkg`. Garanta que a classe `SupervisorNode` herde da classe `Scenario2`:

    ```python
    class SupervisorNode(Node, Scenario2):
        # ...
    ```

    *Obs.: Caso os pacotes não tenham sido construídos com o `--symlink-install`, será necessário construí-los novamente.*

3.  **Ajustar o IP do Servidor e o Modelo LLM:** No arquivo de cenário (`scenario2_llm.py`), localize a função `__init__` da classe `Scenario2` e verifique as variáveis:

    ```python
    # Dentro de Scenario2.__init__
    self.ollama_ip = "192.168.5.150" # <<< VERIFIQUE E AJUSTE ESTE IP
    self.ollama_port = 11434 
    self.ollama_model = "llama3.2" # <<< VERIFIQUE O NOME DO MODELO
    ```

    O `self.ollama_ip` deve ser o endereço IP da máquina onde o servidor Ollama está em execução.

### 3.2. Lançando os Nós (Exemplo: Cenário 2)

É recomendado usar uma ferramenta como o `terminator` para gerenciar múltiplos terminais.

1.  **No Computador da Borda (`edge`):** Abra um terminal e execute o launch file da borda para o Cenário 2.

    ```bash
    ros2 launch edge_launch edge.scn_2.launch.py
    ```

2.  **No Computador Local (`local`):** Abra dois terminais separados.

      * **Terminal 1 (Gazebo):** Lance o cenário no Gazebo.
        ```bash
        ros2 launch local_launch scn_2.launch.py
        ```
      * **Terminal 2 (MPC Local):** Lance os nós do MPC de rastreamento da trajetória.
        ```bash
        ros2 launch local_launch local_2_all.launch.py
        ```

3.  **Iniciando a Simulação no Gazebo:** Após a abertura do Gazebo, clique no botão "play" (canto inferior esquerdo) para que o Gazebo inicie os tópicos do ROS. Neste momento, o Supervisor detectará e registrará os AGVs na GUI.

### 3.3. Interagindo com a Simulação via GUI

  - **Start scenario:** Inicia os procedimentos do cenário. No Cenário 2, o Supervisor controlará os AGVs e a interação com o LLM será ativada.
  - **Stop:** Para o movimento de todos os AGVs.
  - **Set priorities:** Ajusta a prioridade (e velocidade máxima) de cada AGV.
  - **MPC Service:** Permite enviar um AGV para uma posição específica manualmente.

### 3.4. Executando Outros Cenários

Para inicializar os nós dos outros cenários, apenas altere o número do cenário no comando do `launch file`. Por exemplo, para o cenário 1:

  * **Borda:** `ros2 launch edge_launch edge.scn_1.launch.py`
  * **Local (Terminal 1):** `ros2 launch local_launch scn_1.launch.py`
  * **Local (Terminal 2):** `ros2 launch local_launch local_1_all.launch.py`

## 4\. Configurações Adicionais

  * **Parâmetros:** Vários parâmetros como `d_safe`, `d_safe_obs`, `high_vel`, `obstacles`, e `limit_n` (quantidade de AGVs/obstáculos a serem considerados pelo MPC) podem ser ajustados nos `launch files` da borda para otimizar o desempenho. No Cenário 4, a quantidade de AGVs é definida pela variável `n` nos `launch files` locais.
  * **Salvando Dados:** Para salvar dados da simulação, altere a variável `save_sim_data` para `True` nos `launch files` e defina um `sim_name`. Você também precisará configurar a variável `self.path` nos arquivos `mpc_node.py` e `mpc_tracking_node.py` para o diretório onde deseja salvar os dados.

## 5\. Documentos Relacionados

  * [Tese](http://dspace.sti.ufcg.edu.br:8080/jspui/handle/riufcg/30896)
  * [Artigo Elsevier IoT](https://www.sciencedirect.com/science/article/abs/pii/S2542660522001470)
  * [Artigo Congresso Brasileiro de Automática 2022](https://www.sba.org.br/cba2022/wp-content/uploads/artigos_cba2022/paper_9287.pdf)