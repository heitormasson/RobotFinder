# RobotFinder
O presente projeto foi criado como projeto final da disciplina SSC0712 - Sistemas Evolutivos Aplicados à Robótica, e tem por objetivo encontrar um conjunto
de parâmetros (de construção e de movimentação) para um robô manipulador de 3 juntas, que tem por objetivo seguir uma trajetória de pontos previamente determinada.

O vídeo final explicando os algoritmos pode ser encontrado [neste link](https://drive.google.com/drive/folders/19RxmS9Z0M7JJ0cQoU_XLDASArwka2ASw?usp=sharing)

# Descrição dos arquivos
* AG1.ipynb: Jupyter notebook testando e rodando o AG1 (e verificando sua convergência)
* AGS.py: Código principal roda o AG2 para otimizar o AG3, e implementa o AG1 e o AG3 (para serem importados em outros projetos)
* RobotUtils.py: Utiliza a biblioteca roboticstoolbox para criar funções de construção e manipulação de robôs
* ag3_test.ipynb: Testar o ag3 para cada ponto da trajetória, utilizando o robô otimizado pelo AG1, bem como os parâmetros otimizados pelo AG2
* tutorial_roboticstoolbox.ipynb: Breve tutorial a respeito da biblioteca utilizada neste projeto 

# Um pouco de teoria

Um robô manipulador tem como principais características:
* Utilizado para manipular materiais sem contato direto do operador
* Translada pelo espaço tridimensional
* Consegue fazer os movimentos de pitch, roll e spin com o seu efetuador (“garra”)

Exemplos de robôs manipuladores com aplicação industrial estão mostrados na figura abaixo.

![Robos manipuladores](https://github.com/heitormasson/RobotFinder/blob/main/Images/robot_manipulator_examples.png)

A representação matemática de um robô manipulador envolve diferentes sistemas de coordenadas, transformações de rotação e translação, além de uma quantidade razoável de
algebra linear e trignometria. Há, no entanto, uma forma mais simplificada, a partir da **representação de Denavit-Hartenberg (DH)**:
* Representado por um conjunto de juntas, ligadas entre si por meio de elos
* A relação entre elos e juntas pode ser matematicamente expressados pela matriz de Denavit-Hartenberg (DH)
* De forma simplificada, cada junta possui 4 parâmetros: a, ɑ, θ, d.
* Cada junta pode ser prismática (P) ou de rotação (R) 
* Permite encontrar a posição do efetuador a partir da configuração das juntas (cinemática direta)

A imagem abaixo mostra os parâmetros de DH, para uma determinada junta i.

![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/parametros_DH.png)

Um robô com 3 juntas de rotação (RRR) está na figura abaixo:

![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/robot_RRR.png)

E, por outro lado, podemos ter um robô com 3 juntas prismáticas (PPP):

![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/robot_PPP.png)

# Descrição do problema
* Dada uma trajetória de N pontos
* O efetuador do robô deve passar por esses pontos com a menor distância possível
* A trajetória entre pontos subsequentes é aproximada por uma reta
* Problema: 
    *Achar a configuração das juntas a partir de um ponto (cinemática inversa) não é trivial
    * É preciso explorar o espaço de busca das juntas para aproximar o efetuador do ponto desejado -> Algoritmo Evolutivo!
    
# Metodologia

## "AGS que otimizam outros AGS"

* Serão utilizados 3 algoritmos evolutivos
* O “mais externo”, chamado de AG3, é responsável por explorar os parâmetros de movimentação do robô, aproximando-o de um ponto alvo
* O AG2 é responsável por otimizar os parâmetros do AG3, dependendo de qual robô será movimentado
* Dessa forma, um robô RRR terá parâmetros do AG3 diferentes de um robô PPP
* O AG1 é responsável por otimizar os parâmetros de construção do robô para uma determinada trajetória, utilizando para isso o AG3 otimizado

## AG3
1. Técnica de seleção: 
    1. Elitismo (N melhores) 
    1. Torneio de N
1. Técnica de crossover
    1. Média
    1. Melhor entre os pais
    1. Aleatório (50% de probabilidade para cada pai)
1. Mutação
    1. Magnitude da mutação (M)
    1. Fator exponencial (E)
1. Genocídio
    1. Porcentagem do genocídio (P)
    1. Frequência do genocídio (F)



## AG2
* Otimiza um AG3 para determinado robô analisando
* Entrada: objeto robô, parâmetros discretos do AG3 (tipo de seleção e tipo de crossover), ponto inicial e ponto final.
* Indivíduo: Instância do AG3 com diferentes parâmetros contínuos (taxa de mutação, magnitude de mutação, número de indivíduos, decaimento exponencial e passo do decaimento)
* Função objetivo: Recompensa o indivíduo com base na rapidez (do ponto de vista do número de iterações e do tempo por iteração) e com base na variância entre os pontos assumidos pelo manipulador robótico (quanto menor essa variância, maior a pontuação).
* Seleção: Elitismo
* Crossover: Média
* Escolha do melhor: média das últimas 5 gerações
* Saída: Melhores parâmetros a serem utilizados pelo AG3 do Robô analisado.


## AG1
* Utiliza o AG3 já otimizado pelo AG2
* Entrada: Trajetória a ser seguida; parâmetros otimizados do AG3
* Indivíduo: um conjunto de parâmetros de construção do robô
* Função objetivo: Minimizar o erro na execução da trajetória (somatório da diferença entre o ponto atingido e o ponto alvo, para cada ponto)
* Seleção: Elitismo
* Crossover: Média
* Genocídio de 50% a cada 5 iterações 
    * Por se tratar de um problema puramente matemático
    * Saída: Trajetória ideal para o robô em questão

# Fluxograma 

Um fluxograma simplificado da influência de cada AG no sistema final é mostrado abaixo. Temos o AG1 em tom vermelho; AG2 em tom laranja e AG3 em tom azul.

![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/fluxogram_ags.png)

# Resultados

* Desempenho do AG2 para otimização do AG3 para um robô PPP
![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/Resultado_AG2_PPP.png)

* Desempenho do AG3 otimizado

![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/Resultado_Melhor_PPP.png)

* Deslocamento entre dois pontos do Robô PPP utilizando o AG3 otimizado

![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/Deslocamento_melhor_PPP.png)

* Podemos observar o deslocamento de um AG3 não otimizado em uma trajetória com mais pontos e comparar com um AG3 otimizado

![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/deslocamento_nao_otimizado.png)
![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/deslocamento_otimizado.png)

* Observa-se um comportamento claramente mais comportado e linear, sendo que o robô se mexe muito menos para alcançar os pontos desejados.

* Agora, vamos analisar o desempenho de um Robô não otimizado pelo AG1 em comparação com um Robô otimizado para a mesma trajetória.

![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/Non_optimal_robot.png)
![parametros_dh](https://github.com/heitormasson/RobotFinder/blob/main/Images/optimal_robot.png)

* Cada linha do gráfico representa o desempenho do AG3 utilizado para alcançar cada um dos pontos. Vê-se que o robô otimizado alcança os pontos desejados de forma muito mais rápida e eficiente.


