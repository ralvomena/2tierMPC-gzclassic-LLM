# 2tierMPC-gzclassim-LLM

# 2tierMPC-LLM: instruções de execução e simulação

## Visão geral

Este documento descreve os passos para configurar o ambiente e executar os cenários de simulação do projeto 2tierMPC-LLM. Ele cobre a instalação das ferramentas base (ROS 2 e Gazebo), a compilação dos pacotes do projeto e as instruções específicas para executar os diferentes cenários. Os cenários de simulação deste repositório são cópias dos cenários encontrados em [2tierMPC-gzclassic](https://github.com/ralvomena/2tierMPC-gzclassic) e [2tierMPC-igngazebo](https://github.com/ralvomena/2tierMPC-igngazebo), mas com alterações no Cenário 2 e a adição do Cenário 5. O Cenário 2 agora inclui uma LLM (Grande Modelo de Linguagem) para otimização de uma linha de produção hipotética, onde as velocidades dos AGVs são ajustadas para atender a uma determinada taxa de finalização de ordens, medidas em ordens por minuto (OPM). O operador também pode interagir com a LLM via linguagem natural para alterar a OPM ou tratar de um AGV específico, por exemplo. O Cenário 5 é utilizado na validação da LLM no ajuste automático dos pesos da função custo do MPC executado na borda (planejamento de trajetórias). Um AGV é posto para seguir trajetórias definidas por splines, enquanto a LLM analisa os dados das trajetórias executadas e sugere alterações nos pesos Q e R da função custo do MPC.

## 1\. Instalação do ambiente e dependências

As simulações foram validadas com o sistema operacional, ferramentas e pacotes indicados a seguir:

  * Ubuntu 20.04 LTS arm64
  * ROS 2 Foxy Fitzroy
  * Gazebo 11 (Classic)
  * Python 3.8.10
  * CasADi v3.6.7

### 1.1. Ferramentas base (ROS/Gazebo)

A instalação das ferramentas pode ser feita em dois computadores/máquinas virtuais, uma representando o lado da borda e a outra o lado local, ou apenas em um computador/máquina virtual. Se optar por executar em computadores/máquinas virtuais diferentes, instale as ferramentas abaixo em ambos (exceto o Gazebo e o pacote de intregração ROS/Gazebo, instalado apenas no lado local).

1.  **Instalar o ROS 2 Foxy:** siga as instruções detalhadas neste [link](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html) (incluindo o `ros-dev-tools`).

2.  **Configurar o ambiente ROS 2:** para que os comandos do ROS 2 estejam acessíveis, execute no terminal (apenas uma vez):

    ```bash
    echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
    ```

    Veja mais detalhes neste [link](https://docs.ros.org/en/foxy/Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html).

3.  **Instalar o CasADi para Python:** siga as instruções de download e instalação neste [link](https://web.casadi.org/get/).

4.  **Instalar o Gazebo 11 (apenas no lado local):** siga as instruções de instalação neste [link](https://www.google.com/search?q=http://gazebosim.org/tutorials%3Ftut%3Dinstall_ubuntu%26cat%3Dinstall).

5.  **Instalar os pacotes de integração ROS/Gazebo (apenas no lado local):** Execute os comandos:

    ```bash
    sudo apt-get install ros-foxy-gazebo-ros-pkgs
    ```

### 1.2. Ferramentas para integração com a LLM (para os Cenários 2 e 5)

Para executar a funcionalidade de otimização com o LLM, as seguintes ferramentas são necessárias.

1.  **Instalar o Ollama:** Ollama é a ferramenta utilizada para executar LLMs localmente. Siga as instruções de instalação para o seu sistema operacional no site oficial: [ollama.com](https://ollama.com).

2.  **Baixar o modelo LLM:** após instalar o Ollama, baixe o modelo que será utilizado pelo cenário. Exemplo: para utilizar o modelo Phi-3 Mini, abra o terminal e execute:

    ```bash
    ollama pull phi3
    ```

3.  **Instalar o módulo Python `requests`:** para que o nó ROS da simulação possa se comunicar com a API do Ollama, a biblioteca `requests` é utilizada.

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

Para a preparação dos pacotes no lado local, a mesma sequência de comandos deve ser executada a partir do diretório `/local`:

```bash
colcon build # adicione o --sylink-install para alteração do código sem a necessidade de reexecutar o colcon 
echo "source /caminho_ate_o_diretorio/local/install/setup.bash" >> ~/.bashrc
```

**Atenção:** Lembre-se de substituir `/caminho_ate_o_diretorio/` pelo caminho absoluto correto em sua máquina e de abrir um novo terminal para que as alterações tenham efeito.

## 3\. Executando a simulação

O foco na execução é dado aos Cenários 2 e 5, visto que os outros cenários são cópias do projeto base.

### 3.1. Configuração do Cenário 2 (Com LLM)

1.  **Iniciar o Servidor Ollama:** conforme a seção 1.2, garanta que o servidor Ollama esteja rodando e acessível na rede.

2.  **Alterar a classe do Supervisor:** abra o arquivo `supervisor.py` no pacote `edge_tier_pkg`. Garanta que a classe `SupervisorNode` herde da classe `Scenario2`:

    ```python
    class SupervisorNode(Node, Scenario2):
        # ...
    ```

    *Obs.: Caso os pacotes não tenham sido construídos com o `--symlink-install`, será necessário construí-los novamente.*

3.  **Ajustar o IP do servidor e o modelo LLM:** no arquivo de cenário (`scenario2_llm.py`), localize a função `__init__` da classe `Scenario2` e verifique as variáveis:

    ```python
    # Dentro de Scenario2.__init__
    self.ollama_ip = "192.168.5.150" # <<< VERIFIQUE E AJUSTE ESTE IP
    self.ollama_port = 11434 
    self.ollama_model = "phi3" # <<< VERIFIQUE O NOME DO MODELO
    ```

    O `self.ollama_ip` deve ser o endereço IP da máquina onde o servidor Ollama está em execução.

### 3.2. Lançando os nós (Exemplo: Cenário 2)

É recomendado usar uma ferramenta como o `terminator` para gerenciar múltiplos terminais.

1.  **No computador da borda (`edge`):** abra um terminal e execute o launch file da borda para o Cenário 2.

    ```bash
    ros2 launch edge_launch edge.scn_2.launch.py
    ```

2.  **No computador Local (`local`):** abra dois terminais separados.

      * **Terminal 1 (Gazebo):** execute o cenário no Gazebo.
        ```bash
        ros2 launch local_launch scn_2.launch.py
        ```
      * **Terminal 2 (MPC Local):** execute os nós do MPC de rastreamento da trajetória.
        ```bash
        ros2 launch local_launch local_2_all.launch.py
        ```

3.  **Iniciando a Simulação no Gazebo:** após a abertura do Gazebo, o Supervisor detectará e registrará os AGVs na GUI.

### 3.3. Interagindo com a Simulação via GUI

  - **Start scenario:** inicia os procedimentos do cenário. Nos Cenário 2 e 5 a interação com a LLM será ativada.
  - **Stop:** cessa o movimento de todos os AGVs.
  - **MPC Service:** permite chamar o serviço do MPC para mover um AGV para uma posição específica.

### 3.4. Executando outros cenários

Para inicializar os nós dos outros cenários, apenas altere o número do cenário no comando do `launch file`. Por exemplo, para o cenário 5:

  * **Borda:** `ros2 launch edge_launch edge.scn_5.launch.py`
  * **Local (Terminal 1):** `ros2 launch local_launch scn_5.launch.py`
  * **Local (Terminal 2):** `ros2 launch local_launch local_5_all.launch.py`

## 4\. Configurações adicionais

  * **Parâmetros:** vários parâmetros como `d_safe`, `d_safe_obs`, `obstacles`, e `limit_n` (quantidade de AGVs/obstáculos a serem considerados pelo MPC) podem ser ajustados nos `launch files` da borda para otimizar o desempenho. No Cenário 4, a quantidade de AGVs é definida pela variável `n` nos `launch files` locais.
  * **Salvando dados:** para salvar dados da simulação, altere a variável `save_sim_data` para `True` nos `launch files` e defina um `sim_name`. Você também precisará configurar a variável `self.path` nos arquivos `mpc_node.py` e `mpc_tracking_node.py` para o diretório onde deseja salvar os dados.

## 5\. Documentos e projetos relacionados

  * [Projeto base (Gazebo Classic)](https://github.com/ralvomena/2tierMPC-gzclassic)
  * [Migração para o Gazebo Citadel](https://github.com/ralvomena/2tierMPC-igngazebo)
  * [Tese](http://dspace.sti.ufcg.edu.br:8080/jspui/handle/riufcg/30896)
  * [Artigo Elsevier IoT](https://www.sciencedirect.com/science/article/abs/pii/S2542660522001470)
  * [Artigo Congresso Brasileiro de Automática 2022](https://www.sba.org.br/cba2022/wp-content/uploads/artigos_cba2022/paper_9287.pdf)