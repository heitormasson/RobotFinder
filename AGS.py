import numpy as np
# from roboticstoolbox import *
from spatialmath import *
from math import pi
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import random
import RobotUtils
import copy
import multiprocessing

a_lim=[0,15] #parâmetros máximos e mínimos de a
alpha_lim=[0,2*pi] #parâmetros máximos e mínimos de alpha
const_list_r=[0,15] #parâmetros máximos e mínimos para junta de rotação
const_list_p=[0,2*pi] #parâmetros máximos e mínimos para junta prismática
MAX_a = 10
MAX_d = 10
MAX_alpha = np.pi*2
MAX_theta = np.pi*2
CONV_threshold = 0.001 # folga máxima dos pontos da trajetória (3cm)

#função que cria robos
# R para juntas de rotação
# P para juntas prismáticas
# RRR -> Robô com 3 juntas de rotação
robot_build_funcs = {'RRR': RobotUtils.robo_RRR,
                    'RRP' : RobotUtils.robo_RRP,
                    'RPR' : RobotUtils.robo_RPR,
                    'RPP' : RobotUtils.robo_RPP,
                    'PRR' : RobotUtils.robo_PRR,
                    'PRP' : RobotUtils.robo_PRP,
                    'PPR' : RobotUtils.robo_PPR,
                    'PPP' : RobotUtils.robo_PPP}


# funcoes de construcoes a partir dos cromossomos
def birth_robot(cromossomo):
    const_list = [sum([(2 ** j) if cromossomo[j + i] else 0 for j in range(8)]) for i in range(3, 27, 8)]
    a_list = [sum([(2 ** j) if cromossomo[j + i] else 0 for j in range(8)]) for i in range(27, 51, 8)]
    juntas = cromossomo[:3]
    alpha_list = [sum([(2 ** j) if cromossomo[j + i] else 0 for j in range(2)]) for i in range(51, 57, 2)]
    for i in range(3):
        const_list[i] = const_list[i] * MAX_d / 255.0 if juntas[i] else const_list[i] * MAX_theta / 255.0

        alpha_list[i] = alpha_list[i] * MAX_alpha / 4.0
        a_list[i] = a_list[i] * MAX_a / 255.0

    strRobot = ''
    for i in cromossomo[:3]:
        strRobot += 'R' if i else 'P'
    robo = robot_build_funcs[strRobot]('', const_list, a_list, alpha_list)
    return robo, cromossomo[:3]

def point_H(H):
    '''
    Função que retorna um ponto no espaço euclidiano 3D a partir de uma matriz de DH

    :param H: Matriz 4x4 de DH
    :return: array (x,y,z) da posição referente à matriz H
    '''
    return np.array(H.A[0:3, 3])


class AG1:
    def __init__(self, num_crs, juntas_str, juntas_init, trajetoria, ag3_params, mut_rate, co_tech, sel_tech,
                 to_kill_perc, to_kill_freq=1, mut_mag=0.1, exp_decay=0.5, step=10):
        '''
        :param num_crs: (int) numero de cromossomos (individuos) iniciais
        :param juntas_str (str): modelo de robo ('RRR' para 3 juntas de rotação, por exemplo)
        :param Juntas_init (np.array of float): conjunto de parâmetros de movimentação das juntas iniciais
        :param trajetoria (list of np.array): lista, onde cada elemento é um array em 3D, para definir a trajetória inicial
        :param mut_rate (int or dict): porcentagem de individuos que sofrerão mutação
        1) if it is a dict, then: mut_rate={limiar1:mut_rat1,limiar2:mut_rate2 (...) limiarN:mut_rateN}
        - basicallly, if the number of generations without improvement gets to limiar1, then the
        mutation rate is updated to mut_rat1...the same thing applies to mut_rate2...mut_rateN
        2) if it is a int, then the mutation rate of the GA is constant and equals to mut_rate
        :param co_tech (str): técnica de crossover utilizada
        - 'AVG': pega a média (entre os dois pais)
        - 'BST': pega o parâmetro do pai com a maior pontuação
        - 'RAN': pega os parâmetros de um pai aleatoriamente
        :param sel_tech: sel_tech (str) técnica de seleção utilizada neste GA
        - 'EL' para elitismo
        - 'TN' para torneio de N
        :param mut_rate (int or dict): porcentagem de individuos que sofrerão mutação
        1) if it is a dict, then: mut_rate={limiar1:mut_rat1,limiar2:mut_rate2 (...) limiarN:mut_rateN}
        - basicallly, if the number of generations without improvement gets to limiar1, then the
        mutation rate is updated to mut_rat1...the same thing applies to mut_rate2...mut_rateN
        2) if it is a int, then the mutation rate of the GA is constant and equals to mut_rate
        :param to_kill_perc (float from 0 to 1): porcentagem de indivíduos que sofrem o genocídio
        :param to_kill_freq (int >=0): frequência de genocício (se for igual a N, o genocídio acontece a cada N gerações)
        :param mut_mag: magnitude da mutação (o quanto, em porcentagem, o gene pode variar, para mais ou para menos)
        :param exp_decay: aumento exponencial da taxa de mutação
        :param step (int): se for igual a N, a magnitude de mutação exponencialmente a cada N gerações sem um novo melhor de todos

        No fim, temos: gene=gene +- gene*mut_mat*exp_decay**(geracoes_sem_melhora/step)
        '''

        # inicializando os parâmetros
        self.num_crs = num_crs
        self.juntas_init = juntas_init
        self.juntas_str = juntas_str
        self.trajetoria = trajetoria
        self.individuals = []
        self.history_points = []
        self.limits = []
        self.limits.append(a_lim)
        self.limits.append(alpha_lim)
        self.ag3_params = ag3_params

        # verifica o tipo de robo e altera os limites de acordo
        for char in juntas_str:
            if (char == 'R'):
                self.limits.append(const_list_r)
            else:
                self.limits.append(const_list_p)

        # inicializa taxa de mutação
        if (isinstance(mut_rate, dict)):
            self.mut_rate_dict = mut_rate
            self.mut_rate = mut_rate[0]
        else:
            self.mut_rate_dict = None
            self.mut_rate = mut_rate
        # inicializa parâmetros do ag
        self.co_tech = co_tech
        self.sel_tech = sel_tech
        self.to_kill_perc = to_kill_perc
        self.q_gif = []
        self.best_q_gif = []

        self.mut_mag = mut_mag
        self.exp_decay = exp_decay
        self.step = step
        self.to_kill_freq = to_kill_freq

        self.create_new_individuals()

    def dist(self, P):
        '''
        :param P (np.array): ponto em 3D (x,y,z)
        :return: distância entre o ponto P e o ponto P0 (ponto para onde o robô deve ir)
        '''
        return np.abs(np.sqrt((P[0] - self.P0[0]) ** 2 + (P[1] - self.P0[1]) ** 2 + (P[2] - self.P0[2]) ** 2))

    def create_new_individuals(self, initial_conf=None, num_elems=None, delete=False):
        '''
        Cria novos indivíduos

        :param initial_conf (np.array): se for informado, define a configuração inicial (genes) de cada indivíduo
        :param num_elems (int): define o número de novos elementos a serem adicionados
        :param delete (Bool): define se os indivíduos devem ser reiniciados
        :return (list of AG1.individuals: lista com os indivíduos atualizada
        '''
        if (delete == True):  # reinicia a lista de invidíduos
            self.individuals = []
        if (
                num_elems == None):  # se não foi informado o número de elementos, assume-se que ele é igual ao tamanho da população atual
            num_elems = self.num_crs

        try:
            self.best_idx
        except:
            self.best_idx = -1  # caso o melhor de todos ainda não tenha sido configurado

        if (initial_conf == None):  # se nao existe configuração inicial
            for i in range(num_elems):  # para cada elemento
                cr = []
                for gen_idx in range(
                        3):  # para cada gene, crie os parâmetros de construção aleatoriamente, respeitando os limites
                    for in_idx in range(3):
                        cr.append(np.random.uniform(low=self.limits[gen_idx][0], high=self.limits[gen_idx][1]))
                robot = robot_build_funcs[self.juntas_str]('', cr[0:3], cr[3:6],
                                                           cr[6:9])  # crie o robo a partir dos parametros de construção
                self.individuals.append({'robot': robot, 'cr': cr})  # robo e cromossomo armazenados
        else:
            for i in range(num_elems):
                cr = copy.deepcopy(
                    initial_conf)  # caso contrário, crie a partir de uma configuração (cromossomo) informado
                robot = robot_build_funcs[self.juntas_str]('', cr[0:3], cr[3:6], cr[6:9])
                self.individuals.append({'robot': robot, 'cr': cr})

    def evaluate_one(self, ind_idx):  # avalie um indivíduo
        self.individuals[ind_idx]['score'] = 0
        self.individuals[ind_idx]['score_list'] = []
        ag3_total_gens = 100  # defina o numero de gerações do AG3 (a ser rodado para um indivíduo do AG1)
        self.individuals[ind_idx]['trajetoria'] = []
        for idx_traj in range(len(self.trajetoria)):  # rode o AG3 para cada ponto da trajetoria
            if(idx_traj == 0):  # pegue a posição de junta inicial, para a primeira iteração
                juntas_init = self.juntas_init
            else:
                juntas_init = last_solution  # caso o contrário, pegue a ultima solução (último ponto) da iteração anterior

            P0 = self.trajetoria[idx_traj]  # o ponto a ser levado o efetuador é dado pela trajetoria

            # crie o robô
            self.individuals[ind_idx]['robot'] = robot_build_funcs[self.juntas_str]('',
                                                                                    self.individuals[ind_idx]['cr'][
                                                                                    0:3],
                                                                                    self.individuals[ind_idx]['cr'][
                                                                                    3:6],
                                                                                    self.individuals[ind_idx]['cr'][
                                                                                    6:9])

            # crie o AG3 a partir dos parâmetros dados ao AG1 (que foram otimizados pelo AG2)
            ag3 = AG3Int(self.ag3_params['num_ind'], self.individuals[ind_idx]['robot'], juntas_init, P0,
                         self.ag3_params['mut_rate'], self.ag3_params['co_tech'],
                         self.ag3_params['sel_tech'], self.ag3_params['to_kill_perc'], self.ag3_params['to_kill_freq'],
                         self.ag3_params['mut_mag'], self.ag3_params['exp_decay'], self.ag3_params['step'])
            ag3.run(ag3_total_gens)  # rode o AG3
            last_solution = list(ag3.individuals[ag3.best_idx]['cr'])  # salve a ultima solução
            self.individuals[ind_idx]['trajetoria'].append(
                list(ag3.individuals[ag3.best_idx]['cr']))  # salve o ponto na trajetoria

            if (ag3.gen + 1 < ag3_total_gens):  # convergiu
                # pontuação baseada no numero de gerações que demorou para convergir
                self.individuals[ind_idx]['score_list'].append(ag3.gen / ag3_total_gens)  # entre 0 e 1
            else:
                # caso não tenha convergido, o score é baseado no qual próximo o ponto estava, mas com uma penalidade de 1
                # dessa forma, todas as soluções que convergiram são melhores que as que não convergiram
                self.individuals[ind_idx]['score_list'].append(ag3.best_score + 1)  # entre 1 e infinito
        self.individuals[ind_idx]['score'] = sum(self.individuals[ind_idx]['score_list'])

    def evaluate_all(self):  # faça a avaliação individual para cada indivíduo na população
        for i in range(self.num_crs):
            if (i != self.best_idx):
                self.evaluate_one(i)

    def get_mean(self):  # pegue a média da pontuação de todos os indivíduos
        self.mean = np.mean(list(map(lambda x: x['score'], self.individuals)))

    def get_best_individual(self):

        try:
            best_idx_bef = self.best_idx
        except:
            best_idx_bef = -1  # invalido, para mostrar que nao havia sido calculado o best_individual ainda']

        temp_list = list(map(lambda x: x['score'], self.individuals))
        self.best_idx = temp_list.index(min(temp_list))
        self.best_score = min(temp_list)
        self.best_trajetoria = self.individuals[self.best_idx]['trajetoria']

        if (self.best_idx != best_idx_bef):  # verifique se o melhor de todos foi alterado
            self.num_without_imp = 0  # se foi, resete a contagem
        else:
            self.num_without_imp += 1  # se nao, incremente a contagem

    def crossover(self, ind_idx1, ind_idx2, weight1=0.5):
        '''
        Crossover entre dois indivíduos, com indice ind_idx1 e ind_idx2
        :param ind_idx1: indice do primeiro indivíduo (referente a self.individuals)
        :param ind_idx2: indice do segundi indivíduo (referente a self.individuals
        :param weight1: peso dado para escolher o primeiro indivíduo
        '''

        cr1 = list(self.individuals[ind_idx1]['cr'])
        cr2 = list(self.individuals[ind_idx2]['cr'])

        if (self.co_tech == 'AVG'):
            new_ind = [(weight1 * cr1[cr_idx] + (1 - weight1) * cr2[cr_idx]) for cr_idx in
                       range(len(cr1))]  # pegue a média considerando o peso para o indivíduio 1
            if (self.individuals[ind_idx1]['score'] > self.individuals[ind_idx2][
                'score']):  # se o ind 2 for melhor, substitua o 1
                self.individuals[ind_idx1]['cr'] = np.array(new_ind)
            else:  # caso contrário, substitua o 2
                self.individuals[ind_idx2]['cr'] = np.array(new_ind)

        if (self.co_tech == 'BST'):  # substitua o pior (com maior score) pelo melhor
            if (self.individuals[ind_idx1]['score'] < self.individuals[ind_idx2]['score']):
                self.individuals[ind_idx2]['cr'] = np.array(cr1)
            else:
                self.individuals[ind_idx1]['cr'] = np.array(cr2)

        if (self.co_tech == 'RAN'):  # substitua qualquer pai aleatoriamente
            if (ind_idx1 != self.best_idx and ind_idx2 != self.best_idx):
                if (random.random() < 0.5):
                    self.individuals[ind_idx2]['cr'] = np.array(cr1)
                else:
                    self.individuals[ind_idx1]['cr'] = np.array(cr2)

    def selection(self):
        '''
        Selecionar dois indivíduos da população, e executar o crossover entre eles
        '''
        if (self.sel_tech == 'EL'):  # elitismo, o melhor cruza com todos
            for cr_idx in range(self.num_crs):
                if (cr_idx != self.best_idx):  # não cruze o melhor de todos com ele mesmo
                    self.crossover(self.best_idx, cr_idx, 0.75)  # 75% de peso para o melhor de todos
        elif (self.sel_tech[0] == 'T'):  # torneio
            N = int(self.sel_tech[1])  # toreio de 2
            idx_list = random.sample(range(self.num_crs), 2 * N)  # 2N pais aleatórios
            best_sol = 1e9  # inicia com um valor muito alto
            for idx in idx_list[0:N]:
                if (self.individuals[idx]['score'] < best_sol):
                    par1 = idx
                    best_sol = self.individuals[idx][
                        'score']  # pegue o melhor indivíduo do primeiro segmento (N indivíduos)
            best_sol = 1e9
            for idx in idx_list[N:]:
                if (self.individuals[idx]['score'] < best_sol):
                    par2 = idx
                    best_sol = self.individuals[idx][
                        'score']  # pegue o melhor indivíduo do segundo  segmento (N indivíduos)

            self.crossover(par1, par2,
                           0.5)  # faça crossover entre os vencedores do torneio, com peso igual para cada um

        elif (self.sel_tech == 'R'):  # roleta
            lista = []
            score_inv_total = sum(1 / self.individuals[cr_idx]['score'] for cr_idx in range(
                self.num_crs))  # problema de minimizaçaõ -> a probabilidade é maior tal qual menor a pontuação -> fazer a inversa
            for cr_idx in range(self.num_crs):
                try:
                    lista.append(lista[-1] + (
                                1 / self.individuals[cr_idx]['score']) / score_inv_total)  # para os outros indivíduos
                except:
                    lista.append(
                        (1 / self.individuals[cr_idx]['score']) / score_inv_total)  # apenas para o primeiro indivíduo

            prob_list = []  # lista de probabilidade
            for cr_idx in range(self.num_crs):
                try:
                    # a probabilidade é inversamente proporcional ao score
                    prob_list.append(prob_list[-(1 + cr_idx)] + (1 / self.individuals[cr_idx][
                        'score']) / score_inv_total)  # vai somando com a probabilidade anterior (para no fim dar 1)
                except:
                    prob_list.append(
                        (1 / self.individuals[cr_idx]['score']) / score_inv_total)  # apenas para o primeiro indivíduo
            rand_num = random.random()  # pegue um numero aleatorio
            for idx_num in range(len(prob_list)):  # amostre da lista de probabilidade (como se fosse uma distribuição)
                if (rand_num < prob_list[idx_num]): break

            par1 = idx_num  # seleciona o primeiro pai
            rand_num = random.random()  # pegue um numero aleatorio
            for idx_num in range(len(prob_list)):  # amostre da lista de probabilidade (como se fosse uma distribuição)
                if (rand_num < prob_list[idx_num]): break

            par2 = idx_num  # selecione o segundo pai
            self.crossover(par1, par2, 0.5)

    def change_mutation_rate(self):
        self.mut_rate = self.mut_rate_dict[0]
        for key in list(self.mut_rate_dict.keys()):
            if (key > self.num_without_imp):
                break
            self.mut_rate = self.mut_rate_dict[key]

    def mutation(self):
        if (self.mut_rate != 0):  # apenas taxas de mutação não nulas
            for cr_idx in range(self.num_crs):  # percorra todos os indivíduos
                if (cr_idx != self.best_idx):  # não mute o melhor de todos
                    if (
                            random.random() <= self.mut_rate):  # amoestre um número aleatório entre 0 e 1 e verifique se ele é menor que a taxa de mutação
                        # se for, faça a mutação
                        for gen_idx in range(len(self.individuals[cr_idx]['cr'])):  # para cada gene do indivíduo
                            mut_type = (-1) ** np.random.randint(
                                2)  # verifique aleatoriamente se será uma mutação de soma ou subtração
                            self.individuals[cr_idx]['cr'][gen_idx] += mut_type * self.mut_mag * self.exp_decay ** int(
                                self.num_without_imp / self.step)  # aplique a mutação a partir da equação de mutação exponencial

    def kill_worst_elems(self):  # mate os piores indivíduos
        self.individuals.sort(key=lambda x: x['score'])  # faça uma ordenação, para ver quais são os piores
        self.best_idx = 0  # o melhor sempre será o indice 0, já que foi feito uma ordenação
        self.best_score = self.individuals[0]['score']  # salve essa pontuação
        to_kill = int(self.num_crs * self.to_kill_perc)  # defina quantos indivíduos matar
        del self.individuals[-1:-(1 + to_kill):-1]  # mate os indivíduos
        self.create_new_individuals(
            num_elems=to_kill)  # crie um indivíduo novo, para cada indivíduo morto, com gene igual ao do melhor de todos.

    def run(self, num_iters):  # função principal, execute o AG
        self.best_score_list = []  # inicializando melhor de todos
        self.mean_list = []  # inicializando media
        for gen in range(num_iters):  # para cada geracao
            print("Geração {}/{}".format(gen + 1, num_iters))
            self.evaluate_all()
            self.get_best_individual()  # retorna o indice do melhor de todos em self.best_idx

            self.get_mean()  # retorna a media em self.mean
            self.best_score_list.append(self.best_score)
            self.mean_list.append(self.mean)
            if self.best_score < CONV_threshold:  # encerre se ja tiver chego no threshold (3cm)
                break
            self.selection()  # faça a seleção
            if (self.mut_rate_dict != None):  # verifique a atualização da taxa de mutação
                self.change_mutation_rate()
            self.mutation()  # faça a mutação
            if (gen % self.to_kill_freq == 0):  # faça o genocídio, de acordo com a frequência de genocídio
                self.kill_worst_elems()
            print("melhor de todos: {}".format(self.best_score))

class AG3Int:
    def __init__(self, num_crs, robot, Juntas_init, P0, mut_rate, co_tech, sel_tech, to_kill_perc, to_kill_freq=1, mut_mag=0.1, exp_decay=0.5, step=10):
        '''
        :param num_crs: (int) numero de cromossomos (individuos) iniciais
        :param robot (roboticstoolbox Robot): robo a ser utilizado
        :param Juntas_init (np.array of float): conjunto de parâmetros de movimentação das juntas iniciais
        :param P0 (np.array of int):
        :param co_tech (str): técnica de crossover utilizada
        - 'AVG': pega a média (entre os dois pais)
        - 'BST': pega o parâmetro do pai com a maior pontuação
        - 'RAN': pega os parâmetros de um pai aleatoriamente
        :param sel_tech: sel_tech (str) técnica de seleção utilizada neste GA
        - 'EL' para elitismo
        - 'TN' para torneio de N
        :param mut_rate (int or dict): porcentagem de individuos que sofrerão mutação
        1) if it is a dict, then: mut_rate={limiar1:mut_rat1,limiar2:mut_rate2 (...) limiarN:mut_rateN}
        - basicallly, if the number of generations without improvement gets to limiar1, then the
        mutation rate is updated to mut_rat1...the same thing applies to mut_rate2...mut_rateN
        2) if it is a int, then the mutation rate of the GA is constant and equals to mut_rate
        :param to_kill_perc (float from 0 to 1): porcentagem de indivíduos que sofrem o genocídio
        :param to_kill_freq (int >=0): frequência de genocício (se for igual a N, o genocídio acontece a cada N gerações)
        :param mut_mag: magnitude da mutação (o quanto, em porcentagem, o gene pode variar, para mais ou para menos)
        :param exp_decay: aumento exponencial da taxa de mutação
        :param step (int): se for igual a N, a magnitude de mutação exponencialmente a cada N gerações sem um novo melhor de todos

        No fim, temos: gene=gene +- gene*mut_mat*exp_decay**(geracoes_sem_melhora/step)
        '''

        # inicializando os parametros do robo
        self.robot = robot
        self.individuals = []
        self.num_crs = num_crs
        self.P0 = P0
        self.history_points = [] #armazenar o histórico de pontos visitados pelo robô

        if (isinstance(mut_rate, dict)): #configura a taxa de mutação (constante ou variável)
            self.mut_rate_dict = mut_rate
            self.mut_rate = mut_rate[0]
        else:
            self.mut_rate_dict = None
            self.mut_rate = mut_rate

        #parametros gerais do AG
        self.co_tech = co_tech
        self.sel_tech = sel_tech
        self.to_kill_perc = to_kill_perc
        self.initial_pos = Juntas_init
        self.q_gif=[] #variavel que salva um gif das posições das juntas (todos os indivíduos)
        self.best_q_gif=[] #variavel que salva um gif das posições das juntas (melhores indivíduos)
        self.mut_mag=mut_mag
        self.exp_decay=exp_decay
        self.step=step
        self.to_kill_freq=to_kill_freq

        self.create_new_individuals(initial_pos=Juntas_init)

    def dist(self, P):
        '''
        :param P (np.array): ponto em 3D (x,y,z)
        :return: distância entre o ponto P e o ponto P0 (ponto para onde o robô deve ir)
        '''
        return np.abs(np.sqrt((P[0] - self.P0[0]) ** 2 + (P[1] - self.P0[1]) ** 2 + (P[2] - self.P0[2]) ** 2))

    def create_new_individuals(self, initial_pos=None, num_elems=None, delete=False):
        '''
        Cria novos indivíduos

        :param initial_pos (np.array): se for informado, define a posição das juntas (genes) de cada indivíduo
        :param num_elems (int): define o número de novos elementos a serem adicionados
        :param delete (Bool): define se os indivíduos devem ser reiniciados
        :return (list of AG3Int.individuals: lista com os indivíduos atualizada
        '''
        if (delete == True): #reinicia a lista de invidíduos
            self.individuals = []
        if (num_elems == None): #se não foi informado o número de elementos, assume-se que ele é igual ao tamanho da população atual
            num_elems = self.num_crs

        if initial_pos != None: #se não foi dada a posição inicial, criamos uma a partir de um elemento da população
            self.individuals.append({'cr':np.array(initial_pos)})

        for i in range(num_elems): #para cada elemento
            if initial_pos != None:
                if i == 0: #nesse caso, os indivíduos ja estão em ordem decrescente. Então o índice do melhor de todos é 0, necessariamente
                    continue #pule o melhor de todos
                initial_mutation = np.random.randint(3, size=(3,), dtype=int) #defina uma mutação inicial aleatória
                self.individuals.append({'cr': np.array(
                    [float(initial_pos[k] + self.mut_mag*(-1) ** initial_mutation[k]) for k in range(3)])})  # cada indivíduo recebe a posição inicial, porem com uma pequena mutação para mais ou para menos
            else:
                self.individuals.append({'cr': np.random.randint(3600, size=(3,), dtype=int)/10}) #caso não tenha sido dada a posição inicial, crie os indivíduos aleatoriamente

    def evaluate_one(self, ind_idx): #avalie um indivíduo
        H = self.robot.fkine(self.individuals[ind_idx]['cr']) #faça a cinemática direta do robô, com os parâmetros de movimento dados pelo cromossomo.
        P = point_H(H) #pegue a coordenada do ponto a partir da matriz de DH
        self.individuals[ind_idx]['score'] = self.dist(P) #pegue a distância entre o ponto atual e o ponto alvo
        self.history_points.append(self.individuals[ind_idx]['score']) #salve no histórico

    def evaluate_all(self): #faça a avaliação individual para cada indivíduo na população
        for i in range(len(self.individuals)):
            self.evaluate_one(i)

    def get_mean(self): #pegue a média da pontuação de todos os indivíduos
        self.mean = np.mean(list(map(lambda x: x['score'], self.individuals)))

    def get_best_individual(self):
        try:
            best_idx_bef = self.best_idx
        except:
            best_idx_bef = -1  # invalido, para mostrar que nao havia sido calculado o best_individual ainda


        self.best_score = 1e9 #inicie com um valor bem alto
        for i in range(self.num_crs): #verifique o score minimo
            try:
                score = self.individuals[i]['score']
            except:
                self.evaluate_one(i)
                score = self.individuals[i]['score']
            if (score < self.best_score):
                self.best_idx = i
                self.best_score = score
        if (self.best_idx != best_idx_bef): #verifique se o melhor de todos foi alterado
            self.num_without_imp = 0 #se foi, resete a contagem
        else:
            self.num_without_imp += 1 #se nao, incremente a contagem

    def crossover(self, ind_idx1, ind_idx2, weight1=0.5):
        '''
        Crossover entre dois indivíduos, com indice ind_idx1 e ind_idx2
        :param ind_idx1: indice do primeiro indivíduo (referente a self.individuals)
        :param ind_idx2: indice do segundi indivíduo (referente a self.individuals
        :param weight1: peso dado para escolher o primeiro indivíduo
        '''
        cr1 = list(self.individuals[ind_idx1]['cr'])
        cr2 = list(self.individuals[ind_idx2]['cr'])

        if (self.co_tech == 'AVG'):
            new_ind = [(weight1 * cr1[cr_idx] + (1 - weight1) * cr2[cr_idx]) for cr_idx in range(len(cr1))] #pegue a média considerando o peso para o indivíduio 1
            if (self.individuals[ind_idx1]['score'] > self.individuals[ind_idx2]['score']): #se o ind 2 for melhor, substitua o 1
                self.individuals[ind_idx1]['cr'] = np.array(new_ind)
            else: #caso contrário, substitua o 2
                self.individuals[ind_idx2]['cr'] = np.array(new_ind)

        if (self.co_tech == 'BST'):
            if (self.individuals[ind_idx1]['score'] < self.individuals[ind_idx2]['score']): #substitua o pior (com maior score) pelo melhor
                self.individuals[ind_idx2]['cr'] = np.array(cr1)
            else:
                self.individuals[ind_idx1]['cr'] = np.array(cr2)

        if (self.co_tech == 'RAN'): #substitua qualquer pai aleatoriamente
            if(ind_idx1!=self.best_idx and ind_idx2!=self.best_idx):
                if (random.random() < 0.5):
                    self.individuals[ind_idx2]['cr'] = np.array(cr1)
                else:
                    self.individuals[ind_idx1]['cr'] = np.array(cr2)

    def selection(self):
        '''
        Selecionar dois indivíduos da população, e executar o crossover entre eles
        '''
        if (self.sel_tech == 'EL'): #elitismo, o melhor cruza com todos
            for cr_idx in range(self.num_crs):
                if (cr_idx != self.best_idx):  # não cruze o melhor de todos com ele mesmo
                    self.crossover(self.best_idx, cr_idx, 0.75)  # 75% de peso para o melhor de todos
        elif (self.sel_tech[0] == 'T'):  # torneio
            N = int(self.sel_tech[1])  # torneio de 2
            idx_list = random.sample(range(self.num_crs), 2 * N)  # 2N pais aleatórios
            best_sol = 1e9 #iniciando com um valor muito alto
            for idx in idx_list[0:N]:
                if (self.individuals[idx]['score'] < best_sol):
                    par1 = idx
                    best_sol=self.individuals[idx]['score'] #pegue o melhor indivíduo do primeiro segmento (N indivíduos)
            best_sol = 1e9 #iniciando com um valor muito alto
            for idx in idx_list[N:]:
                if (self.individuals[idx]['score'] < best_sol):
                    par2 = idx
                    best_sol = self.individuals[idx]['score'] #pegue o melhor indivíduo do segundo segmento (N indivíduos)

            self.crossover(par1, par2, 0.5) #faça crossover entre os vencedores do torneio, com peso igual para cada um

        elif (self.sel_tech == 'R'): # roleta
            lista = []
            score_inv_total = sum(1 / self.individuals[cr_idx]['score'] for cr_idx in range(self.num_crs)) # problema de minimizaçaõ -> a probabilidade é maior tal qual menor a pontuação -> fazer a inversa
            for cr_idx in range(self.num_crs):
                try:
                    lista.append(lista[-1] + (1 / self.individuals[cr_idx]['score']) / score_inv_total) #para os outros indivíduos
                except:
                    lista.append((1 / self.individuals[cr_idx]['score']) / score_inv_total) #apenas para o primeiro indivíduo

            prob_list = [] #lista de probabilidade
            for cr_idx in range(self.num_crs):
                try:
                    # a probabilidade é inversamente proporcional ao score
                    prob_list.append(prob_list[-(1+cr_idx)] + (1 / self.individuals[cr_idx]['score']) / score_inv_total) #vai somando com a probabilidade anterior (para no fim dar 1)
                except:
                    prob_list.append((1 / self.individuals[cr_idx]['score']) / score_inv_total) #apenas para o primeiro indivíduo
            rand_num = random.random() #pegue um numero aleatorio
            for idx_num in range(len(prob_list)): #amostre da lista de probabilidade (como se fosse uma distribuição)
                if (rand_num < prob_list[idx_num]): break

            par1=idx_num #seleciona o primeiro pai
            rand_num = random.random() #pegue um numero aleatorio
            for idx_num in range(len(prob_list)): #amostre da lista de probabilidade (como se fosse uma distribuição)
                if (rand_num < prob_list[idx_num]): break

            par2=idx_num #selecione o segundo pai
            self.crossover(par1,par2,0.5)

    def mutation(self): #faça a mutação
        
        if (self.mut_rate != 0): #apenas taxas de mutação não nulas
            mutation_mag= max(self.mut_mag*self.exp_decay**int(self.num_without_imp/self.step),1e-6)
            for cr_idx in range(self.num_crs): #percorra todos os indivíduos
                if (cr_idx != self.best_idx): #não mute o melhor de todos
                    if (random.random() <= self.mut_rate): #amostre um número aleatório entre 0 e 1 e verifique se ele é menor que a taxa de mutação
                        #se for, faça a mutação
                        for gen_idx in range(len(self.individuals[cr_idx]['cr'])): #para cada gene do invidíduo
                            mut_type = (-1)**np.random.randint(2) #verifique aleatóriamente se será uma mutação de soma ou subtração
                            self.individuals[cr_idx]['cr'][gen_idx] += mut_type*mutation_mag #aplique a mutação a partir da equação de mutação exponencial

    def kill_worst_elems(self): #mate os piores indivíduos
        if (len(self.individuals) >= 3): #evite matar se a população for muito pequena
            self.individuals.sort(key=lambda x: x['score']) # faça uma ordenação, para ver quais são os piores
            self.best_idx = 0 #o melhor sempre será o indice 0, já que foi feito uma ordenação
            self.best_score = self.individuals[0]['score'] #salve essa pontuação
            to_kill = int(self.num_crs * self.to_kill_perc-1) #defina quantos indivíduos matar
            del self.individuals[-1:-(1 + to_kill):-1] #mate os indivíduos
            self.create_new_individuals(num_elems=to_kill, initial_pos=list(self.individuals[0]['cr'])) #crie um indivíduo novo, para cada indivíduo morto, com gene igual ao do melhor de todos.

    def save_q_gif(self): #salve as variáveis de junta (cromossomos) de cada indivíduo da população, para poder ser mostrado em um gif posteriormente
        for cr_idx in range(int(self.num_crs)):
            self.q_gif.append(self.individuals[cr_idx]['cr'])

    def save_best_q_gif(self): #salve as variáveis de junta (cromossomos) do melhor indivíduo atual da população, para poder ser mostrado em um gif posteriormente
        self.best_q_gif.append(self.individuals[self.best_idx]['cr'])

    def video_q_gif(self): #mostre cada indivíduo da população em um gif
        self.robot.plot(np.array(self.q_gif), movie='all_individuals.gif', dt=0.01)
    def video_best_gif(self): #mostre a lista dos melhores de todos em um gif
        self.robot.plot(np.array(self.best_q_gif), movie='best_individual.gif',dt=0.01)

    def run(self, num_iters): #função principal, execute o AG
        self.best_score_list = [] #inicializando melhor de todos
        self.mean_list = [] #inicializando media
        for gen in range(num_iters): #para cada geracao
            self.gen=gen
            self.evaluate_all()
            self.get_best_individual()  # retorna o indice do melhor de todos em self.best_idx
            self.get_mean()  # retorna a media em self.mean
            self.best_score_list.append(self.best_score) #salve na lista
            self.mean_list.append(self.mean) #salve na lista
            if self.best_score < CONV_threshold: #encerre se ja tiver chego no threshold (3cm)
                break
            self.save_q_gif() #salve os cromossomos
            # self.save_best_q_gif() #salve o melhor cromossomo
            self.selection() #faça a seleção
            self.mutation() #faça a mutação
            if(gen%self.to_kill_freq == 0): #faça o genocídio, de acordo com a frequência de genocídio
                self.kill_worst_elems()

    def plot_q_gif(self,cmap): #faça um scatter plot dos indivíduos, cujo cmap é ponderado pela geração do indivíduo
        x_data = []
        y_data = []
        z_data = []
        gen_data = []
        gen_num = 0
        gen = 0
        for q_conf in self.q_gif:
            if (gen_num == self.num_crs):
                gen += 1
                gen_num = 0
            [x, y, z] = self.robot.fkine(q_conf).A[0:3, 3]
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)
            gen_data.append(gen)
            gen_num += 1
        fig, axes = plt.subplots()
        axes = plt.axes(projection='3d')
        axes.scatter3D(x_data, y_data, z_data, c=gen_data, cmap=cmap)
        axes.scatter3D(self.P0[0], self.P0[1], self.P0[2], c='k', s=70)
        # plt.savefig('Movement')

    def plot_best_q_gif(self,cmap): #faça um scatter plot dos melhores indivíduos, cujo cmap é ponderado pela geração
        x_data = []
        y_data = []
        z_data = []
        gen_data = []
        gen_num = 0
        gen = 0
        for q_conf in self.best_q_gif:
            if (gen_num == self.num_crs):
                gen += 1
                gen_num = 0
            [x, y, z] = self.robot.fkine(q_conf).A[0:3, 3]
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)
            gen_data.append(gen)
            gen_num += 1
        fig, axes = plt.subplots()
        axes = plt.axes(projection='3d')
        axes.scatter3D(x_data, y_data, z_data, c=gen_data, cmap=cmap)
        axes.scatter3D(self.P0[0], self.P0[1], self.P0[2], c='k', s=70)

class AG2continuous:
    '''
    Optimizes AG3 continuous parameters
    cromossome = {n of individuals, mut rate, to kill perc, mut_mag, exp_decay, step}
    '''
    def __init__(self, num_crs, robot, Pinit, P0, sel_tech, co_tech, mut_mag, mut_rate, initial_cr=None):
        '''
        :param num_crs: numero de individuos do AG2
        :param robot: objeto robo para o qual sera otimizado um AG3
        :param Pinit: posicao inicial da qual partirao os individuos AG3
        :param P0: posicao final para onde os AG3 deverao convergir
        :param sel_tech: tecnica de selecao utilizada pelo AG3
        :param co_tech: tecnica de crossover utilizada pelo AG3
        :param mut_rate: taxa de mutacao do AG2
        :param mut_mag: magnitude de mutacao do AG2
        :initial_cr: cromossomo inicial
        '''

        self.robot_arch = robot
        self.initial_pos = Pinit
        self.P0 = P0
        self.num_crs = num_crs
        self.sel_tech = sel_tech
        self.co_tech = co_tech

        self.best_idx = 0
        self.current_best_idx = 0
        self.best_score = 0

        self.individuals = []

        self.num_without_imp = 0

        self.mut_rate = mut_rate
        self.mut_mag = mut_mag
        self.to_kill_freq=to_kill_freq

        self.best_score_par = []

        self.score_history = [[] for i in range(num_crs)]

        self.initial_cr = initial_cr


        self.create_new_individuals(initial_cr=initial_cr)


    def create_new_individuals(self, initial_cr=None):
        """
        cromossome = {n of individuals, mut rate, to kill perc, mut_mag, exp_decay, step}
        """
        if initial_cr == None:
            for i in range(self.num_crs):
                cromossome = np.random.uniform(size=6)
                self.individuals.append({'cr': cromossome})
                self.individuals[-1]['ind'] = self.generate_individual(cromossome)
        else:
            for i in range(self.num_crs):
                self.individuals.append({'cr': initial_cr})
                self.individuals[-1]['ind'] = self.generate_individual(initial_cr)

    def generate_individual(self, cromossome):
        num_ind = int(cromossome[0]*47)+3  # [3,50]
        mut_rate = cromossome[1]*0.5 + 0.5  # [0.5, 1]
        to_kill_perc = cromossome[2]  # [0,1]
        mut_mag = cromossome[3]*0.5+0.0001   # [0.0001,0.5] 
        exp_decay = cromossome[4]   #  [0, 1]
        step = int(cromossome[5]*99)+1   # [1, 100]

        return AG3Int(num_ind, self.robot_arch, self.initial_pos, self.P0, mut_rate, self.co_tech,
                        self.sel_tech, to_kill_perc, mut_mag=mut_mag, exp_decay=exp_decay, step=step)
        
    def refresh_individuals(self, not_chaged_list=[]):
        for idx in range(self.num_crs):
            ind = self.individuals[idx]
            if idx == self.best_idx or idx in not_chaged_list or idx == self.current_best_idx:
                continue
            ind['ind'] = self.generate_individual(ind['cr'])

    def evaluate_one(self, indx, score_result):
        """
        :param indx: index of the individual to be evaluated
        :param score_result: variable to be stored the evaluation result
        """

        ind = self.individuals[indx]
        score= 0
        t0 = time.time() # Pega o tempo inicial da execucao do AG3
        ind['ind'].run(500)
        t1 = time.time()# Pega o tempo final da execucao do AG3
        
        iterations = len(ind['ind'].best_score_list)
        last_point_distance = ind['ind'].best_score_list[-1]
        enlapsed_time = t1-t0

        # Caso o ponto final alcancado esteja dentro do limiar de convergencia estabelecido, ganha a pontuacao total
        # referente a convergencia
        if last_point_distance <= CONV_threshold: 
            score_conv = 10
        else:
            # Caso contrario, ganha uma quantidade inversamente proporcional a distancia
            score_conv = 10*(CONV_threshold/last_point_distance)
        
        score += score_conv

        # Pontuacao referente ao numero de iteracoes utilizadas. Menos de 50 resulta na pontuacao total, caindo proporcionalmente
        # conforme o numero de iteracoes aumenta
        score_it = min((500-iterations)/450, 1)*10
        score += score_it

        # Pontuacao dada proprcional ao tempo medio decorrido em cada iteracao. Quanto menor, melhor        
        score_time = max((10e-3 - enlapsed_time/iterations)/10e-3, 0)

        mean_window_var = 0

        # Calculo da variancia entre os pontos de uma mesma geracao do AG3
        for i in range(0, iterations*ind['ind'].num_crs, ind['ind'].num_crs):
            mean_window_var += np.var(ind['ind'].history_points[i:i+ind['ind'].num_crs])

        # Media das variancias por geracao
        mean_window_var = mean_window_var/iterations

        # Pontuacao proporcional atribuida a variancia
        if mean_window_var < CONV_threshold:
            score_var = mean_window_var*1/CONV_threshold
        elif mean_window_var < 0.05:
            score_var = 1
        elif mean_window_var > 0.05 and mean_window_var < 0.08:
            score_var = 2 - 20*mean_window_var
        elif mean_window_var > 0.08 and mean_window_var < 1:
            score_var = 0.4348 - 0.4348*mean_window_var
        else:
            score_var = 0

        # Pontuacao da variancia vale mais que as anteriores, mas eh ponderada pelas pontuacoes referentes a convergencia
        score_var = score_var*(score_conv/10)*np.sqrt(score_it/10)
        score += score_var*15

        # Calculo da diferenca entre a pontuacao media e a do melhor de todos
        score_mean = np.array(ind['ind'].best_score_list) - np.array(ind['ind'].mean_list)
        score_mean = np.abs(score_mean)
        score_mean = sum(score_mean)

        # Atribui uma pontuacao proporcional
        if score_mean < 0.05:
            score_mean = 1
        elif score_mean < 8:
            score_mean = np.exp(1/(score_mean+1.3927))-1
        else:
            score_mean = 0

        # Pondera a media pelas pontuacoes de convergencia
        score_mean = score_mean*(score_conv/10)*np.sqrt(score_it/10)

        score += score_mean*10

        score_result.value = score


    def evaluate_all(self):
        '''
        Avalia cada individuo paralelamente utilizando a biblioteca multiprocessing
        '''
        jobs = []

        ret_values = []
        for idx in range(0, self.num_crs):
            ret_values.append(multiprocessing.Value("d", 0.0, lock=False))
            p = multiprocessing.Process(target=self.evaluate_one, args=(idx, ret_values[-1]))
            jobs.append(p)
            p.start()

        # Atribui a pontuacao referente a cada individuo e tambem acumula na lista historica
        for j in range(self.num_crs):
            jobs[j].join()
            self.individuals[j]['score'] = ret_values[j].value
            self.score_history[j].append(ret_values[j].value)
            if len(self.score_history[j])>5:
                self.score_history[j].pop(0)

    def get_mean(self):
        self.mean = np.mean(list(map(lambda x: x['cr'], self.individuals)))

    def get_best_individual(self):
        """
        Seleciona o melhor individuo com base na media historica e tambem o melhor individuo atual
        """
        try:
            best_idx_bef = self.best_idx
        except:
            best_idx_bef = -1  # invalido, para mostrar que nao havia sido calculado o best_individual ainda

        temp_list = list(map(lambda x:x['score'],self.individuals))
        temp_list2 = list(map(lambda x:sum(x)/len(x),self.score_history))

        self.best_idx = temp_list2.index(max(temp_list2))
        self.current_best_idx = temp_list.index(max(temp_list))

        self.best_score = temp_list[self.best_idx]
        if (self.best_idx != best_idx_bef):
            self.num_without_imp = 0
        else:
            self.num_without_imp += 1
 
        print('_______________________')
        for i in range(self.num_crs):
            print(self.individuals[i]['cr'])
            print(temp_list[i])
            print('\n')

        print('best: %i'%self.best_idx)
        print('current best: %i'%self.current_best_idx)

    def crossover(self, ind_idx1, ind_idx2, weight1):
        cr1 = list(self.individuals[ind_idx1]['cr'])
        cr2 = list(self.individuals[ind_idx2]['cr'])        
        new_ind = [(weight1 * cr1[cr_idx] + (1 - weight1) * cr2[cr_idx]) for cr_idx in range(len(cr1))]

        self.individuals[ind_idx2]['cr'] = np.array(new_ind)

    def selection(self):
        for cr_idx in range(self.num_crs):
            if (cr_idx != self.best_idx and cr_idx != self.current_best_idx):  # Nao transa caso seja o melhor geral ou o melhor atual
                self.crossover(self.best_idx, cr_idx, 0.75)  # 75% de peso para o melhor de todos

    def mutation(self):
        if (self.mut_rate):                            
            for cr_idx in range(self.num_crs):
                if (cr_idx != self.best_idx and cr_idx != self.current_best_idx):
                    for gen_idx in range(len(self.individuals[cr_idx]['cr'])):
                        if (random.random() <= self.mut_rate):
                            mutation_mag = self.mut_mag
                        
                            if self.num_without_imp>2 and self.num_without_imp<5:
                                mutation_mag = mutation_mag/self.num_without_imp
                            elif self.num_without_imp>5:
                                mutation_mag = min(mutation_mag*(self.num_without_imp/5)**2, 0.15)

                            mut_type = (-1)**np.random.randint(2)
                            self.individuals[cr_idx]['cr'][gen_idx] += mut_type*mutation_mag
                            
                            if self.individuals[cr_idx]['cr'][gen_idx] >1:
                                self.individuals[cr_idx]['cr'][gen_idx] = 1
                            elif self.individuals[cr_idx]['cr'][gen_idx] <0:
                                self.individuals[cr_idx]['cr'][gen_idx] = 0

    def run(self, num_iters):
        self.best_score_list = []
        self.mean_list = []
        for gen in range(num_iters):
            print("Generation: %i/%i"%(gen, num_iters))
            self.evaluate_all()
            self.get_best_individual()  # retorna o indice do melhor de todos em self.best_idx
            self.get_mean()  # retorna a media em self.mean
            self.best_score_list.append(self.best_score)
            self.mean_list.append(self.mean)
            if gen == num_iters-1:
                break
            self.selection()
            self.mutation()
            self.refresh_individuals()


def generate_AG3_from_cromossome(cromossome, robot_arch, initial_pos, P0, co_tech, sel_tech):
    num_ind = int(cromossome[0]*47)+3  # [3,50]
    mut_rate = cromossome[1]*0.5 + 0.5  # [0.5, 1]
    to_kill_perc = cromossome[2]  # [0,1]
    mut_mag = cromossome[3]*0.5+0.0001  # [0.001,2] 
    exp_decay = cromossome[4]   #  [0, 1]
    step = int(cromossome[5]*99)+1   # [1, 100]
    try:
        to_kill_freq = int(cromossome[6]*74)+1 # [1,75]
    except:
        to_kill_freq=1
    print(num_ind)
    print(mut_rate)
    print(to_kill_freq)
    print(to_kill_perc)
    print(mut_mag)
    print(exp_decay)
    print(step)
    return AG3Int(num_ind, robot_arch, initial_pos, P0, mut_rate, co_tech,
                    sel_tech, to_kill_perc, to_kill_freq=to_kill_freq, mut_mag=mut_mag, exp_decay=exp_decay, step=step)

def evalAG3(ind):
        score= 0
        t0 = time.time()
        ind.run(500)
        t1 = time.time()
        # print('ran '+str(indx))
        iterations = len(ind.best_score_list)
        last_point_distance = ind.best_score_list[-1]
        enlapsed_time = t1-t0

        if last_point_distance <= CONV_threshold:
            score_conv = 10
        else:
            score_conv = 10*(CONV_threshold/last_point_distance)
        
        score += score_conv

        score_it = min((500-iterations)/450, 1)*10
        score += score_it

        # score += 0.01*(1.27e-3 - enlapsed_time/iterations)/1.27e-5
        
        score_time = max((10e-3 - enlapsed_time/iterations)/10e-3, 0)

        mean_window_var = 0

        for i in range(0, iterations*ind.num_crs, ind.num_crs):
            mean_window_var += np.var(ind.history_points[i:i+ind.num_crs])

        mean_window_var = mean_window_var/iterations

        if mean_window_var < CONV_threshold:
            score_var = mean_window_var*1/CONV_threshold
        elif mean_window_var < 0.5:
            score_var = 1
        elif mean_window_var > 0.5 and mean_window_var < 1:
            score_var = 2 - 2*mean_window_var
        else:
            score_var = 0

        if(score_conv>=1 and score_it>=1):
            score+=10*score_var
            if(score_var>0):
                score+=2.5*score_time

        return score, (score_time,score_conv, score_it, score_var)


if __name__ == "__main__":

    const_list = [5,5,5]
    a_list = [2,2,2]
    alpha_list = [np.pi/2,np.pi/2,np.pi/2]

    sel_tech='EL' # 'EL' or 'TN' ou 'R' OBS: so acompanha a media de forma rigida se for elitimos (isso ja era esperado)
    co_tech='AVG' # 'AVG' or 'BST' or 'RAN'

    for robType in robot_build_funcs:
        const_list = [5,5,5]
        a_list = [2,2,2]
        alpha_list = [np.pi/2,np.pi/2,np.pi/2]
        robot = robot_build_funcs[robType]('', const_list, a_list, alpha_list)

        print('Built random robot')

        H = robot.fkine([4,3,4])
        P0=point_H(H)
        print('Got initial pos')

        # Posicao inicial das juntas (seria o ponto anterior da trajetoria)
        juntas_init=[0,0,0]

        # Inicializa o AG2 para a combinacao de juntas atual
        ag2 = AG2continuous(5,robot,juntas_init,P0,co_tech,sel_tech,0.05, 1)

        print('Initialized AG2, running...')
        t1= time.time()
        ag2.run(100)
        print(time.time()-t1)
        print('best for: '+robType)
        print(ag2.individuals[ag2.best_idx])
        plt.plot(ag2.mean_list, label='Media')
        plt.plot(ag2.best_score_list, label='Melhor')
        plt.title("Resultado AG2 "+robType)
        plt.legend()
        plt.ylabel('Score')
        plt.xlabel('Iteracoes')
        plt.savefig('Resultado_AG2_'+robType+'.png')
        plt.clf()
        best_ag3 = ag2.individuals[ag2.best_idx]
        cr = best_ag3['cr']
        ag3 = generate_AG3_from_cromossome(cr, robot, juntas_init, P0, co_tech, sel_tech)
        ag3.run(1000)
        f = open('best'+robType+'.txt', 'w')
        f.write(str(best_ag3['cr']))
        f.close()
        plt.plot(ag3.mean_list, label='Media')
        plt.plot(ag3.best_score_list, label='Melhor')
        plt.ylabel('Distancia do objetivo')
        plt.xlabel('Iteracoes')
        plt.title('Resultado Melhor AG3')
        plt.savefig('Resultado_Melhor_'+robType+'.png')
        plt.clf()
        ag3.plot_q_gif(cmap='Reds')
        plt.title('Deslocamento')
        plt.savefig('Deslocamento_melhor_'+robType+'.png')
        plt.clf()

