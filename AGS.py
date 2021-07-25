import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import *
from spatialmath import *
from math import pi
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import random
import RobotUtils
import copy

MAX_a = 10
MAX_d = 10
MAX_alpha = np.pi*2
MAX_theta = np.pi*2
CONV_threshold = 0.01

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
    #     for i in range(2, 57, 8):
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


def move_robot(robot, cromossomo, juntas_bool):
    '''
    Parameters:
    :param robot (roboticstoolbox Robot)
    :param cromossomo (bool de 3 bytes): cada byte representa uma junta
    :param juntas_bool (bool de 3 bits): cada bool representa o tipo de junta:
    - True para junta de rotação
    - False para junta prismática

    Output:

    '''
    q_list = [sum([(2 ** j) if cromossomo[j + i] else 0 for j in range(8)]) for i in range(0, 24, 8)]
    num_juntas = len(juntas_bool)

    q_list = [(q_list[i] / 255) * MAX_theta if juntas_bool[i] else (q_list[i] / 255) * MAX_d for i in range(num_juntas)]
    H = robot.fkine(q_list)

    return H


def point_H(H):
    return np.array(H.A[0:3, 3])


class AG3Int:
    def __init__(self, num_crs, robot, boolJuntas, Juntas_init, P0, mut_rate, co_tech, sel_tech, to_kill_perc, mut_mag=0.1, exp_decay=0.5, step=10):
        '''
        Parameters:
        :param co_tech (str): technique of crossover used in this GA
        - 'OP' for one point
        - 'MPX' for multipoint, with X cross points
        - 'U' for uniform
        :param sel_tech (str) technique of selection used in this GA
        - 'EL' for elitism
        - 'TN' for N-branch tournament selection
        :param mut_rate (int or dict): number of genes to mutate over each new solution (after selection)

        1) if it is a dict, then: mut_rate={limiar1:mut_rat1,limiar2:mut_rate2 (...) limiarN:mut_rateN}
        - basicallly, if the number of generations without improvement gets to limiar1, then the
        mutation rate is updated to mut_rat1...the same thing applies to mut_rate2...mut_rateN

        2) if it is a int, then the mutation rate of the GA is constant and equals to mut_rate
        '''
        # parametros do robo
        self.robot = robot
        self.boolJuntas = boolJuntas
        self.individuals = []
        self.num_crs = num_crs
        self.P0 = P0

        self.history_points = []

        if (isinstance(mut_rate, dict)):
            self.mut_rate_dict = mut_rate
            self.mut_rate = mut_rate[0]
        else:
            self.mut_rate_dict = None
            self.mut_rate = mut_rate

        self.co_tech = co_tech
        self.sel_tech = sel_tech
        self.to_kill_perc = to_kill_perc
        self.initial_pos = Juntas_init
        self.q_gif=[]
        self.best_q_gif=[]

        self.mut_mag=mut_mag
        self.exp_decay=exp_decay
        self.step=step

        self.create_new_individuals(ini0tial_pos=Juntas_init)

    def dist(self, P):
        return np.abs(np.sqrt((P[0] - self.P0[0]) ** 2 + (P[1] - self.P0[1]) ** 2 + (P[2] - self.P0[2]) ** 2))

    def create_new_individuals(self, initial_pos=None, individual_list=None, num_elems=None, delete=False):
        if (delete == True):
            self.individuals = []
        if (num_elems == None):
            num_elems = self.num_crs

        if initial_pos != None:
            self.individuals.append({'cr':np.array(initial_pos)})
        for i in range(num_elems):
            if (individual_list == None):
                if initial_pos != None:
                    if i == 0:
                        continue
                    initial_mutation = np.random.randint(3, size=(3,), dtype=int)
                    self.individuals.append({'cr': np.array(
                        [float(initial_pos[k] + self.mut_mag*(-1) ** initial_mutation[k]) for k in range(3)])})  # precisao de 0.1 grau
                else:
                    self.individuals.append({'cr': np.random.randint(3600, size=(3,), dtype=int)/10})
            else:
                if self.initial_pos != None:
                    initial_mutation = np.random.randint(3, size=(3,), dtype=int)
                    self.individuals.append(
                        {'cr': np.array([self.initial_pos[k] + ((-1) ** initial_mutation[k])*self.init_mut_mag for k in range(3)])})

                else:
                    self.individuals.append({'cr': (np.random.randint(3600, size=(3,), dtype=int))/10})

    def evaluate_one(self, ind_idx):
        P = self.robot.fkine(self.individuals[ind_idx]['cr']).A[0:3, 3]
        self.individuals[ind_idx]['score'] = self.dist(P)
        self.history_points.append(self.individuals[ind_idx]['score'])

    def evaluate_all(self):
        for i in range(self.num_crs):
            self.evaluate_one(i)

    def get_mean(self):
        self.mean = np.mean(list(map(lambda x: x['score'], self.individuals)))

    def get_best_individual(self):
        
        try:
            best_idx_bef = self.best_idx
        except:
            best_idx_bef = -1  # invalido, para mostrar que nao havia sido calculado o best_individual ainda

        self.best_idx = self.individuals.index(min(self.individuals, key=lambda x: x['score']))
        self.best_score = self.individuals[self.best_idx]

        # temp_list = list(map(lambda x:x['score'],self.individuals))        
        # self.best_idx = temp_list.index(min(temp_list))


        # self.best_score = 1e9


        # for i in range(self.num_crs):
        #     try:
        #         score = self.individuals[i]['score']
        #     except:
        #         self.evaluate_one(i)
        #         score = self.individuals[i]['score']
        #     if (score < self.best_score):
        #         self.best_idx = i
        #         self.best_score = score
        if (self.best_idx != best_idx_bef):
            self.num_without_imp = 0
        else:
            self.num_without_imp += 1

    def crossover(self, ind_idx1, ind_idx2, weight1=0.5):
        '''
        Crossover between two individuals, with index ind_idx1 and ind_idx2.

        Parameters:
        :param ind_idx1 (int): index of the first individual (as in self.individuals list)
        :param ind_idx2 (int): index of the second individual (as in self.individuals list)
        '''
        cr1 = list(self.individuals[ind_idx1]['cr'])
        cr2 = list(self.individuals[ind_idx2]['cr'])

        if (self.co_tech == 'AVG'):
            new_ind = [(weight1 * cr1[cr_idx] + (1 - weight1) * cr2[cr_idx]) for cr_idx in range(len(cr1))]
            if (self.individuals[ind_idx1]['score'] > self.individuals[ind_idx2]['score']):
                self.individuals[ind_idx1]['cr'] = np.array(new_ind)
            else:
                self.individuals[ind_idx2]['cr'] = np.array(new_ind)

        if (self.co_tech == 'BST'):
            if (self.individuals[ind_idx1]['score'] < self.individuals[ind_idx2]['score']):
                self.individuals[ind_idx2]['cr'] = np.array(cr1)
            else:
                self.individuals[ind_idx1]['cr'] = np.array(cr2)

        if (self.co_tech == 'RAN'):
            if(ind_idx1!=self.best_idx and ind_idx2!=self.best_idx):
                if (random.random() < 0.5):
                    self.individuals[ind_idx2]['cr'] = np.array(cr1)
                else:
                    self.individuals[ind_idx1]['cr'] = np.array(cr2)

    def selection(self):
        # self.evaluate_all()
        # self.get_best_individual()  # retorna o indice do melhor de todos em self.best_idx
        # self.get_mean()  # retorna a media em self.mean
        if (self.sel_tech == 'EL'):
            for cr_idx in range(self.num_crs):
                if (cr_idx != self.best_idx):  # nao sei se tem algum jeito mais esperto do que testar todos
                    self.crossover(self.best_idx, cr_idx, 0.75)  # 75% de peso para o melhor de todos
                    # self.mutation
        elif (self.sel_tech[0] == 'T'):  # tournament
            N = int(self.sel_tech[1])  # tournament of N
            idx_list = random.sample(range(self.num_crs), 2 * N)  # parents at random
            best_sol = 1e9
            for idx in idx_list[0:N]:
                if (self.individuals[idx]['score'] < best_sol):
                    par1 = idx

            best_sol = 1e9
            for idx in idx_list[N:]:
                if (self.individuals[idx]['score'] < best_sol):
                    par2 = idx

            self.crossover(par1, par2, 0.5)

        elif (self.sel_tech == 'R'): # roulette
            lista = []
            score_inv_total = sum(1 / self.individuals[cr_idx]['score'] for cr_idx in range(self.num_crs))
            for cr_idx in range(self.num_crs):
                try:
                    lista.append(lista[-1] + (1 / self.individuals[cr_idx]['score']) / score_inv_total)
                except:
                    lista.append((1 / self.individuals[cr_idx]['score']) / score_inv_total)

            prob_list = []
            for cr_idx in range(self.num_crs):
                try:
                    prob_list.append(prob_list[-1] + (1 / self.individuals[cr_idx]['score']) / score_inv_total)
                except:
                    prob_list.append((1 / self.individuals[cr_idx]['score']) / score_inv_total)
            rand_num = random.random()
            for idx_num in range(len(prob_list)):
                if (rand_num < prob_list[idx_num]): break

            par1=idx_num
            rand_num = random.random()
            for idx_num in range(len(prob_list)):
                if (rand_num < prob_list[idx_num]): break

            par2=idx_num
            self.crossover(par1,par2,0.5)

    def change_mutation_rate(self):
        self.mut_rate = self.mut_rate_dict[0]
        for key in list(self.mut_rate_dict.keys()):
            if (key > self.num_without_imp):
                break
            self.mut_rate = self.mut_rate_dict[key]

    def mutation(self):
        # self.evaluate_all()
        # self.get_best_individual()  # retorna o indice do melhor de todos em self.best_idx
        
        if (self.mut_rate != 0):
            for cr_idx in range(self.num_crs):
                if (cr_idx != self.best_idx):
                    if (random.random() <= self.mut_rate):
                        for gen_idx in range(len(self.individuals[cr_idx]['cr'])):
                            # self.individuals[cr_idx]['cr'][gen_idx] += float(np.random.normal(loc=0, scale=0.5)*self.individuals[cr_idx]['cr'][gen_idx])
                            
                            self.individuals[cr_idx]['cr'][gen_idx] += self.mut_mag*self.exp_decay**int(self.num_without_imp/self.step)

    def kill_worst_elems(self):
        self.individuals.sort(key=lambda x: x['score'])
        to_kill = int(self.num_crs * self.to_kill_perc)
        del self.individuals[-1:-(1 + to_kill):-1]
        self.create_new_individuals(num_elems=to_kill, initial_pos=self.individuals[self.best_idx])

    def save_q_gif(self, gen,best_perc=1):
        self.individuals.sort(key=lambda x: x['score'])
        for cr_idx in range(int(best_perc*self.num_crs)):
            self.q_gif.append(self.individuals[cr_idx]['cr'])

    def save_best_q_gif(self):
        self.best_q_gif.append(self.individuals[self.best_idx]['cr'])

    def plot_q_gif(self):
        self.robot.plot(np.array(self.q_gif), movie='all_individuals.gif', dt=0.01)
    def plot_best_gif(self):
        self.robot.plot(np.array(self.best_q_gif), movie='best_individual.gif',dt=0.01)
    def run(self, num_iters):
        self.best_score_list = []
        self.mean_list = []
        for gen in range(num_iters):
            self.evaluate_all()
            self.get_best_individual()  # retorna o indice do melhor de todos em self.best_idx
            if self.best_score < CONV_threshold:
                break
            self.get_mean()  # retorna a media em self.mean
            self.best_score_list.append(self.best_score)
            self.mean_list.append(self.mean)
            # self.save_q_gif(gen)
            # self.save_best_q_gif()
            self.selection()
            if (self.mut_rate_dict != None):
                self.change_mutation_rate()
            self.mutation()
            self.kill_worst_elems()
            # print(self.num_without_imp,self.mut_rate)

    def plot_q_gif(self,cmap):
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

    def plot_best_q_gif(self,cmap):
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

"""
class AG3:
    def __init__(self, num_crs, robot, boolJuntas, P0, mut_rate, co_tech, sel_tech, to_kill_perc):
        '''
        Parameters:
        :param co_tech (str): technique of crossover used in this GA
        - 'OP' for one point
        - 'MPX' for multipoint, with X cross points
        - 'U' for uniform
        :param sel_tech (str) technique of selection used in this GA
        - 'EL' for elitism
        - 'TN' for N-branch tournament selection
        :param mut_rate (int or dict): number of genes to mutate over each new solution (after selection)
        1) if it is a dict, then: mut_rate={limiar1:mut_rat1,limiar2:mut_rate2 (...) limiarN:mut_rateN}
        - basicallly, if the number of generations without improvement gets to limiar1, then the
        mutation rate is updated to mut_rat1...the same thing applies to mut_rate2...mut_rateN

        2) if it is a int, then the mutation rate of the GA is constant and equals to mut_rate
        :param boolJuntas (bool de 3 bits): cada bool representa o tipo de junta:
        - True para junta de rotação
        - False para junta prismática
       '''
        # parametros do robo
        self.robot = robot
        self.boolJuntas = boolJuntas
        self.individuals = []
        self.num_crs = num_crs
        self.P0 = P0
        if (isinstance(mut_rate, dict)):
            self.mut_rate_dict = mut_rate
            self.mut_rate = mut_rate[0]
        else:
            self.mut_rate_dict = None
            self.mut_rate = mut_rate

        self.co_tech = co_tech
        self.sel_tech = sel_tech
        self.to_kill_perc = to_kill_perc

        self.create_new_individuals()

    def pos_from_move(self, ind_idx):
        '''
        Parameters:
        :param robot (roboticstoolbox Robot)
        :param cromossomo (bool de 3 bytes): cada byte representa uma junta


        Output:

        '''
        q_list = [sum([(2 ** j) if self.individuals[ind_idx]['cr'][j + i] else 0 for j in range(8)]) for i in
                  range(0, 24, 8)]
        num_juntas = len(self.boolJuntas)

        q_list = [(q_list[i] / 255) * MAX_theta if self.boolJuntas[i] else (q_list[i] / 255) * MAX_d for i in
                  range(num_juntas)]
        H = self.robot.fkine(q_list)

        return np.array(H.A[0:3, 3])

    def dist(self, P):
        return np.abs(np.sqrt((P[0] - self.P0[0]) ** 2 + (P[1] - self.P0[1]) ** 2 + (P[2] - self.P0[2]) ** 2))

    def create_new_individuals(self, individual_list=None, num_elems=None, delete=False):
        if (delete == True):
            self.individuals = []
        if (num_elems == None):
            num_elems = self.num_crs
        for i in range(num_elems):
            if (individual_list == None):
                self.individuals.append({'cr': np.random.randint(2, size=(24,), dtype=bool)})
            else:
                self.individuals.append({'cr': individual_list[i]})

    def evaluate_one(self, ind_idx):
        P = self.pos_from_move(ind_idx)
        self.individuals[ind_idx]['score'] = self.dist(P)

    def evaluate_all(self):
        for i in range(self.num_crs):
            self.evaluate_one(i)

    def get_mean(self):
        self.mean = np.mean(list(map(lambda x: x['score'], self.individuals)))

    def get_best_individual(self):
        try:
            best_idx_bef = self.best_idx
        except:
            best_idx_bef = -1  # invalido, para mostrar que nao havia sido calculado o best_individual ainda

        self.best_score = 1e9
        for i in range(self.num_crs):
            try:
                score = self.individuals[i]['score']
            except:
                self.evaluate_one(i)
                score = self.individuals[i]['score']
            if (score < self.best_score):
                self.best_idx = i
                self.best_score = score
        if (self.best_idx != best_idx_bef):
            self.num_without_imp = 0
        else:
            self.num_without_imp += 1

    def crossover(self, ind_idx1, ind_idx2, bias_1):
        '''
        Crossover between two individuals, with index ind_idx1 and ind_idx2.

        Parameters:
        :param ind_idx1 (int): index of the first individual (as in self.individuals list)
        :param ind_idx2 (int): index of the second individual (as in self.individuals list)
        :param bias_1 (float): bias to select the genes from the first individual.
        - If bias_1=0.75, there is a 75% chance of the child getting its cromossomes from individual 1
        (and, consequently 25% chance to get it from individual 2)
        '''
        cr1 = list(self.individuals[ind_idx1]['cr'])
        cr2 = list(self.individuals[ind_idx2]['cr'])
        new_ind = []

        if (self.co_tech[0:2] == 'MP'):
            num_cp = int(self.co_tech[2])
            cps = list(np.random.randint(0, 24, num_cp))
            cps.append(0)  # getting the first cut on the beginning of the cromossome
            cps.append(24)  # getting the last cut on the end of the cromossome
            cps.sort()

            for cp_idx in range(num_cp + 1):
                if (random.random() < bias_1):  # select the first individual
                    # print("individuo 1")
                    new_ind += cr1[cps[cp_idx]:cps[cp_idx + 1]]
                else:
                    # print("individuo 2")
                    new_ind += cr2[cps[cp_idx]:cps[cp_idx + 1]]

        elif (self.co_tech == 'OP'):
            cps = [0, random.randint(0, 23), 24]  # selecting the crosspoint
            new_ind = []
            for cp_idx in range(2):  # only one crosspoint (thus, two segments)
                if (random.random() < bias_1):
                    new_ind += cr1[cps[cp_idx]:cps[cp_idx + 1]]
                else:
                    new_ind += cr2[cps[cp_idx]:cps[cp_idx + 1]]

        elif (self.co_tech == 'U'):
            for point in range(24):
                if (random.random() < bias_1):  # select the first individual
                    # print("individuo 1")
                    new_ind.append(cr1[point])
                else:
                    # print("individuo 2")
                    new_ind.append(cr2[point])

        if (self.individuals[ind_idx1]['score'] > self.individuals[ind_idx2]['score']):
            self.individuals[ind_idx1]['cr'] = np.array(new_ind)
        else:
            self.individuals[ind_idx2]['cr'] = np.array(new_ind)

    def selection(self):
        self.evaluate_all()
        self.get_best_individual()  # retorna o indice do melhor de todos em self.best_idx
        self.get_mean()  # retorna a media em self.mean
        if (self.sel_tech == 'EL'):
            for cr_idx in range(self.num_crs):
                if (cr_idx != self.best_idx):  # nao sei se tem algum jeito mais esperto do que testar todos
                    self.crossover(self.best_idx, cr_idx, 0.75)  # 75% de chance de pegar o melhor de todos
                    self.mutation
        elif (self.sel_tech[0] == 'T'):  # tournament
            N = int(self.sel_tech[1])  # tournament of N
            idx_list = random.sample(range(self.num_crs), 2 * N)  # parents at random
            best_sol = 1e9
            for idx in idx_list[0:N]:
                if (self.individuals[idx]['score'] < best_sol):
                    par1 = idx

            best_sol = 1e9
            for idx in idx_list[N:]:
                if (self.individuals[idx]['score'] < best_sol):
                    par2 = idx

            self.crossover(par1, par2, 0.5)

    def change_mutation_rate(self):
        self.mut_rate = self.mut_rate_dict[0]
        for key in list(self.mut_rate_dict.keys()):
            if (key > self.num_without_imp):
                break
            self.mut_rate = self.mut_rate_dict[key]


    def mutation(self):
        if (self.mut_rate != 0):
            for cr_idx in range(self.num_crs):
                gens_to_mutate = np.random.randint(0, 24, self.mut_rate)
                for gen_to_mutate in list(gens_to_mutate):
                    if (cr_idx != self.best_idx):
                        self.individuals[cr_idx]['cr'][gen_to_mutate] = not self.individuals[cr_idx]['cr'][
                            gen_to_mutate]

    def kill_worst_elems(self):
        self.individuals.sort(key=lambda x: x['score'])
        to_kill = int(self.num_crs * self.to_kill_perc)
        del self.individuals[-1:-(1 + to_kill):-1]
        self.create_new_individuals(num_elems=to_kill)
        pass

    def run(self, num_iters):
        self.best_score_list = []
        self.mean_list = []
        for gen in range(num_iters):
            self.selection()
            if (self.mut_rate_dict != None):
                self.change_mutation_rate()
            self.mutation()
            self.kill_worst_elems()
            self.best_score_list.append(self.best_score)
            self.mean_list.append(self.mean)
            # print(self.num_without_imp,self.mut_rate)
"""

class AG2:
    """
    Optmizes AG3 parameters: 
    """
    def __init__(self, ind_number, ):
        pass

class AG2continuous:
    """
    Optimizes AG3 continuous parameters
    
    cromossome = {n of individuals, mut rate, to kill perc, ut_mag, exp_decay, step}
    """
    def __init__(self, num_crs, robot, boolJuntas, Pinit, P0, sel_tech, co_tech):

        self.robot_arch = robot
        self.boolJuntas = boolJuntas
        self.initial_pos = Pinit
        self.P0 = P0
        self.num_crs = num_crs
        self.sel_tech = sel_tech
        self.co_tech = co_tech

        self.best_idx = 0
        self.best_score = 0

        self.individuals = []


        self.create_new_individuals()

        for mut_rate in mut_rate_list:
            num_iters=1000 #numero de iteracoes
            num_inds=10 #numero de individuos
            sel_tech='T4' # 'EL' or 'TN' OBS: so acompanha a media de forma rigida se for elitimos (isso ja era esperado)
            co_tech='AVG' # 'AVG' or 'BST' or 'RAN'
            #mut_rate={0:1,5:2,10:3,15:4,50:5,100:10,200:20}
            #mut_rate=0.05
            ag3_int = AG3Int(num_inds,robot,boolJuntas,juntas_init,P1,mut_rate,co_tech,sel_tech,0)
            ag3_int.run(num_iters)


    def create_new_individuals(self):
        """
        cromossome = {int, int, int}
                 {n of individuals, mut rate, to kill perc, ut_mag, exp_decay, step}
        """
        for i in range(self):
            cromossome = np.random.uniform(size=6)
            self.individuals.append({'cr': cromossome})
            self.individuals[-1]['ind'] = self.generate_individual(cromossome)

    def generate_individual(self, cromossome):
        num_ind = int(cromossome[0]*49)+1  # [1,50]
        mut_rate = cromossome[1]  # [0,1]
        to_kill_perc = cromossome[2]  # [0,1]
        mut_mag = cromossome[3]*2   # [0,2] 
        exp_decay = cromossome[4]   #  [0, 1]
        step = int(cromossome[5]*99)+1   # [1, 100]
        return AG3Int(num_ind, self.robot_arch, self.boolJuntas, self.initial_pos, self.P0, mut_rate, self.co_tech,
                        self.sel_tech, to_kill_perc, mut_mag=mut_mag, exp_decay=exp_decay, step=step)
    def refresh_individuals(self, not_chaged_list=[]):
        for ind in self.individuals:
            idx= self.individuals.index(ind)
            if idx == self.best_idx or idx in not_chaged_list:
                continue
            ind['ind'] = self.generate_individual(ind['cr'])

    def evaluate_one(self, ind):
        """
        Peso da avaliacao:
        0.2 - se convergiu antes do limite
        0.2*(CONV_threshol/last_best_score) - else

        min((limite-iteracoes)/450, 1)*0.2

        0.01*(1.27e-3 - enlapsed_time/iterations)/1.27e-5 - Cada porcento de alteracao no tempo medio de convergencia resulta em 
                                                            adicao ou remocao  de 0.01 da nota

        0.4*CONV_threshold*4/var(points_distances)

        """
        score= 0

        t0 = time.time()
        ind['ind'].run(500)
        t1 = time.time()

        iterations = len(ind['ind'].best_score_list)
        last_point_distance = ind['ind'].best_score_list[-1]
        enlapsed_time = t1-t0

        if last_point_distance <= CONV_threshold:
            score += 0.2
        else:
            score += 0.2*(CONV_threshold/last_point_distance)
            score += min((limite-iteracoes)/450, 1)*0.2

        score += 0.01*(1.27e-3 - enlapsed_time/iterations)/1.27e-5

        score += 0.4*CONV_threshold*4/np.var(points_distances)        

        self.ind['score'] = score

    def evaluate_all(self):
        for ind in self.individuals:
            self.evaluate_one(ind)

    def get_best_individual(self):
        try:
            best_idx_bef = self.best_idx
        except:
            best_idx_bef = -1  # invalido, para mostrar que nao havia sido calculado o best_individual ainda

        self.best_idx = self.individuals.index(min(self.individuals, key=lambda x: x['score']))
        self.best_score = self.individuals[self.best_idx]

        # temp_list = list(map(lambda x:x['score'],self.individuals))        
        # self.best_idx = temp_list.index(min(temp_list))

        if (self.best_idx != best_idx_bef):
            self.num_without_imp = 0
        else:
            self.num_without_imp += 1

    def crossover(self, ind_idx1, ind_idx2, weight1):
        cr1 = list(self.individuals[ind_idx1]['cr'])
        cr2 = list(self.individuals[ind_idx2]['cr'])
        
        new_ind = [(weight1 * cr1[cr_idx] + (1 - weight1) * cr2[cr_idx]) for cr_idx in range(len(cr1))]
        if (self.individuals[ind_idx1]['score'] > self.individuals[ind_idx2]['score']):
            self.individuals[ind_idx1]['cr'] = np.array(new_ind)
        else:
            self.individuals[ind_idx2]['cr'] = np.array(new_ind)

    def selection(self):
        for cr_idx in range(self.num_crs):
            if (cr_idx != self.best_idx):  # nao sei se tem algum jeito mais esperto do que testar todos
                self.crossover(self.best_idx, cr_idx, 0.75)  # 75% de peso para o melhor de todos

    def mutation(self):
        if (self.mut_rate):
            for cr_idx in range(self.num_crs):
                if (cr_idx != self.best_idx):
                    if (random.random() <= self.mut_rate):
                        for gen_idx in range(len(self.individuals[cr_idx]['cr'])):
                            # self.individuals[cr_idx]['cr'][gen_idx] += float(np.random.normal(loc=0, scale=0.5)*self.individuals[cr_idx]['cr'][gen_idx])
                            self.individuals[cr_idx]['cr'][gen_idx] += self.mut_mag*self.exp_decay**int(self.num_without_imp/self.step)

    def kill_worst_elems(self):
        pass

    def run(self):
        self.best_score_list = []
        self.mean_list = []
        for gen in range(num_iters):
            self.evaluate_all()
            self.get_best_individual()  # retorna o indice do melhor de todos em self.best_idx
            self.get_mean()  # retorna a media em self.mean
            self.best_score_list.append(self.best_score)
            self.mean_list.append(self.mean)
            self.selection()
            self.mutation()
            self.refresh_individuals()


if __name__ == "__main__":
    pass