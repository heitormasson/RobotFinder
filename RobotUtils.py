import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import *
from spatialmath import *
from math import pi
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import cm
import random
matplotlib notebook


#definindo uma formula geral para um robo RRR (3 juntas)
def robo_RRR(my_name,const_list,a_list,alpha_list):
    robot = DHRobot(
      [
        RevoluteDH(d=const_list[0],a=a_list[0],alpha=alpha_list[0],offset=0),
        RevoluteDH(d=const_list[1],a=a_list[1],alpha=alpha_list[1],offset=0),
        RevoluteDH(d=const_list[2],a=a_list[2],alpha=alpha_list[2],offset=0),
       ], name=my_name)
    return robot

#fazendo o robo especifico acima (chamado de robo antropomorfico)
#d1 = 10, só como exemplo

def create_robo_ant(d1,name):
    const_list=[d1,0,0]
    a_list=[0,0,0]
    alpha_list=[pi/2,0,0]
    return robo_RRR(name,const_list,a_list,alpha_list) 


def robo_RRP(my_name,const_list,a_list,alpha_list):
    robot = DHRobot(
      [
        RevoluteDH(d=const_list[0],a=a_list[0],alpha=alpha_list[0],offset=0),
        RevoluteDH(d=const_list[1],a=a_list[1],alpha=alpha_list[1],offset=0),
        PrismaticDH(a=a_list[2],alpha=alpha_list[2],theta=const_list[2],offset=0),
       ], name=my_name)
    return robot

def create_robo_scara(d1,r2,d3,name):
    const_list=[d1,0,0]
    a_list=[0,r2,0]
    alpha_list=[0,pi,0]
    return robo_RRP(name,const_list,a_list,alpha_list)  

def robo_RPR(my_name,const_list,a_list,alpha_list):
    robot = DHRobot(
      [
        RevoluteDH(d=const_list[0],a=a_list[0],alpha=alpha_list[0],offset=0),
        PrismaticDH(a=a_list[1],alpha=alpha_list[1],theta=const_list[1],offset=0),
        RevoluteDH(d=const_list[2],a=a_list[2],alpha=alpha_list[2],offset=0),

       ], name=my_name)
    return robot

def robo_RPP(my_name,const_list,a_list,alpha_list):
    robot = DHRobot(
      [
        RevoluteDH(d=const_list[0],a=a_list[0],alpha=alpha_list[0],offset=0),
        PrismaticDH(a=a_list[1],alpha=alpha_list[1],theta=const_list[1],offset=0),
        PrismaticDH(a=a_list[2],alpha=alpha_list[2],theta=const_list[2],offset=0),
       ], name=my_name)
    return robot

def robo_PRR(my_name,const_list,a_list,alpha_list):
    robot = DHRobot(
      [
        PrismaticDH(a=a_list[0],alpha=alpha_list[0],theta=const_list[0],offset=0),
        RevoluteDH(d=const_list[1],a=a_list[1],alpha=alpha_list[1],offset=0),
        RevoluteDH(d=const_list[2],a=a_list[2],alpha=alpha_list[2],offset=0),
       ], name=my_name)
    return robot

def robo_PRP(my_name,const_list,a_list,alpha_list):
    robot = DHRobot(
      [
        PrismaticDH(a=a_list[0],alpha=alpha_list[0],theta=const_list[0],offset=0),
        RevoluteDH(d=const_list[1],a=a_list[1],alpha=alpha_list[1],offset=0),
        PrismaticDH(a=a_list[2],alpha=alpha_list[2],theta=const_list[2],offset=0),
       ], name=my_name)
    return robot

def robo_PPR(my_name,const_list,a_list,alpha_list):
    robot = DHRobot(
      [
        PrismaticDH(a=a_list[0],alpha=alpha_list[0],theta=const_list[0],offset=0),
        PrismaticDH(a=a_list[1],alpha=alpha_list[1],theta=const_list[1],offset=0),
        RevoluteDH(d=const_list[2],a=a_list[2],alpha=alpha_list[2],offset=0),
       ], name=my_name)
    return robot

def robo_PPP(my_name,const_list,a_list,alpha_list):
    robot = DHRobot(
      [
        PrismaticDH(a=a_list[0],alpha=alpha_list[0],theta=const_list[0],offset=0),
        PrismaticDH(a=a_list[1],alpha=alpha_list[1],theta=const_list[1],offset=0),
        PrismaticDH(a=a_list[2],alpha=alpha_list[2],theta=const_list[2],offset=0),
       ], name=my_name)
    return robot

def dist(P1,P2): 
    return np.abs(np.sqrt(P1[0]**2+P1[1]**2+P1[2]**2)-np.sqrt(P2[0]**2+P2[1]**2+P2[2]**2))



