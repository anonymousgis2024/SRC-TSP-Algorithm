import copy
import math
import os
import random
import time
import elkai
import mutations
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial as ssp
from geopy.distance import geodesic

def readDataFile_list(fileName):
    dataList = list()
    must_points = list()
    types = list()
    type_n = list()
    with open(r"../dataset/real_data/"+fileName,'r') as fr:
        fr.readline()
        for line in fr:
            strs = line.strip().split(",")
            dataList.append((float(strs[1]),float(strs[2])))
            if int(strs[3])==0:
                must_points.append(int(strs[0]))
            else:
                if int(strs[3]) in type_n:
                    types[int(strs[3])-1].append(int(strs[0]))
                else:
                    type_n.append(int(strs[3]))
                    type = list()
                    type.append(int(strs[0]))
                    types.append(type)
    return dataList,must_points,types

def get_add_points(m,types):
    add_points = []
    while len(add_points)<m:
        a_p = []
        for i in range(len(types)):
            r = random.randint(0,len(types[i])-1)
            p = types[i][r]
            a_p.append(p)
        if a_p not in add_points:
            add_points.append(a_p)
    return add_points

class gt:
    def __init__(self,M,O):
        self.M = M
        self.O = O
        self.position = self.M+self.O
        random.shuffle(self.position)
        self.fitness = float('inf')

def get_road_fitness(road,dis_mat):
    r = copy.deepcopy(road)
    s = r[0]
    r.append(s)
    dis = 0
    for i in range(1,len(r)):
        dis += dis_mat[r[i-1]][r[i]]
    return dis

def get_fitness(zq):
    fitness = []
    for x in zq:
        fitness.append(1/x.fitness)
    return fitness

def selection(fitness, num):
    def select_one(fitness, fitness_sum):
        size = len(fitness)
        i = random.randint(0, size - 1)
        while True:
            if random.random() < fitness[i] / fitness_sum:
                return i
            else:
                i = (i + 1) % size
    res = set()
    fitness_sum = sum(fitness)
    while len(res) < num:
        t = select_one(fitness, fitness_sum)
        res.add(t)
    return res

def get_choose_city(list1,list2):
    list = []
    for i in range(len(list1)):
        a = random.randint(0, 1)
        if a==0:
            list.append(list1[i])
        elif a==1:
            list.append(list2[i])
    return list

def crossover(parent1, parent2):
    a = random.randint(1, len(parent1) - 1)
    child1, child2 = parent1[:a], parent2[:a]

    for i in range(len(parent1)):
        if parent2[i] not in child1:
            child1.append(parent2[i])

        if parent1[i] not in child2:
            child2.append(parent1[i])

    return child1, child2

def sort_by_fitness(zq):
    new_zq = []
    zq_id = []
    zq_fitness = []
    for i in range(len(zq)):
        zq_id.append(i)
        zq_fitness.append(zq[i].fitness)
    id_list = [x for _, x in sorted(zip(zq_fitness, zq_id))]
    for j in id_list:
        new_zq.append(zq[j])
    return new_zq


def aberrance(parent3,o3,types):
    old = copy.deepcopy(o3)
    r = random.randint(0,len(o3)-1)
    r2 = random.randint(0,len(types[r])-1)
    o3[r] = types[r][r2]
    for i in range(len(parent3)):
        if parent3[i] in old:
            index = old.index(parent3[i])
            parent3[i] = o3[index]
    return parent3,o3

def select_next_zq(zq,init_zq,m):
    zq_next = zq[:m]
    return zq_next

def draw_points_type(points,types):
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    n = len(types)
    for i, (x, y) in enumerate(points):
        plt.scatter(x, y, color='k', s=3)
        if i in types[0]:
            plt.scatter(x,y,color = 'b',s = 15)
        elif i in types[1]:
            plt.scatter(x, y, color='g', s=15)
        elif i in types[2]:
            plt.scatter(x, y, color='y', s=15)
        elif i in types[3]:
            plt.scatter(x, y, color='k', s=15)
        elif i in types[4]:
            plt.scatter(x, y, color='r', s=15)
        plt.text(x, y, str(i), fontsize=6, color='b')

def plt_edges(edges_touple, points, color):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    for p1,p2 in edges_touple:
        plt.plot([points[p1][0],points[p2][0]], [points[p1][1],points[p2][1]], color=color, linewidth=1)

def get_distmat(M):
    length = M.shape[0]
    distmat = np.zeros((length, length))
    for i in range(length):
        for j in range(i + 1, length):
            distmat[i][j] = distmat[j][i] = np.linalg.norm(M[i] - M[j])
    return distmat

def LKH(city_num,od_dis_mx):
    distance = np.zeros((len(city_num), len(city_num)))
    for i in range(len(city_num)):
        for j in range(len(city_num)):
            distance[i][j] = od_dis_mx[city_num[i]][city_num[j]]
    def cal_fitness(sol):
        tot_len = np.sum(distance[sol[:-1], sol[1:len(sol)]])
        return tot_len

    sol = elkai.solve_float_matrix(distance,runs=10)
    sol.append(0)
    sumdis = cal_fitness(sol)
    return sol, sumdis

def get_same_type_city(t,types):
    for i in range(len(types)):
        if t in types[i]:
            r = random.randint(0,len(types[i])-1)
            return types[i][r]
    else:
        return -1

def turn_points(best_points,points):
    r = []
    for i in range(len(best_points)):
        x = best_points[i][0]
        y = best_points[i][1]
        for j in range(len(points)):
            p_x = points[j][0]
            p_y = points[j][1]
            if x==p_x and y == p_y:
                r.append(j)
                break
    return r
def turn_tour(best_tour,points_r):
    pt = []
    t = copy.deepcopy(best_tour)
    for e in t:
        new_e = [points_r[e[0]],points_r[e[1]]]
        pt.append(new_e)
    return pt

def distance(points,i,j):
    if i!=j:
        p1_x,p1_y = copy.deepcopy(points[i])
        pi = tuple([p1_x-90,p1_y])
        p2_x,p2_y = copy.deepcopy(points[j])
        pj = tuple([p2_x-90,p2_y])
        dis = geodesic(pi, pj).m
        return dis
    else:
        return float('inf')

def count_len(tour,C):
    l = 0
    for edge in tour:
        f = edge[0]
        t = edge[1]
        add = C[f][t]
        l +=add
    return l

def read_dis_mx(file,dis_mx):
    with open(r"../dataset/real_data/"+file,'r') as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            atrs = line.split(",")
            o1,d1,dis = int(atrs[0]),int(atrs[1]),float(atrs[2])
            dis_mx[o1-1][d1-1]=dis
    return dis_mx

def GA_LKH_main(file1,file2,n,time_limit):
    """
    :param file1:     This file records the coordinate information and semantics of each point
    :param file2:     This file records the distance of the road network between any two points
    :param n:         Maximum number of iterations
    :param time_limit:      Upper limit for algorithm execution
    """
    m = 25   # The size of the genetic sample
    v = 0.2  # mutation factor
    points,must_points,types = readDataFile_list(file1) #By reading file 1, the location information and semantic attributes of the node are obtained
    p_n = len(points)

    C = np.zeros((p_n, p_n))
    C = read_dis_mx(file2, C)   #By reading file2, the road network distance between any two points is obtained, and the distance matrix is obtained
    np.random.seed(0)
    zq = []
    add_points = get_add_points(m,types)    #m genetic samples were obtained
    for a in add_points:
        l = gt(must_points,a)               #Generate a genetic sample
        tpn = l.position
        sol, sumdis = LKH(tpn,C)
        l.fitness = sumdis
        zq.append(l)
    fitness_list = get_fitness(zq)

    init_zq = copy.deepcopy(zq)
    start_time = time.time()
    """
    genetic evolution
    The algorithm is not allowed to iterate more than n times.
    """
    for i in range(n):
        now = time.time()
        time_flag = now-start_time
        """If the algorithm runs longer than time_limit, it stops"""
        if time_flag>time_limit:
            print(f"Time spent = {time_flag}s")
            break
        for j in range(m):
            p1, p2 = selection(fitness_list, 2)     #Select two genetic samples
            parent1 = copy.deepcopy(zq[p1].position)
            parent2 = copy.deepcopy(zq[p2].position)
            o1 = copy.deepcopy(zq[p1].O)
            o2 = copy.deepcopy(zq[p2].O)

            new_choose_city = get_choose_city(o1, o2)
            for k in range(len(parent1)):
                if parent1[k] in o1:
                    index = o1.index(parent1[k])
                    parent1[k] = new_choose_city[index]
                if parent2[k] in o2:
                    index = o2.index(parent2[k])
                    parent2[k] = new_choose_city[index]

            new_choose_city_1 = copy.deepcopy(new_choose_city)
            new_choose_city_2 = copy.deepcopy(new_choose_city)

            # Start mutating operation
            if random.random() < v:
                r_1 = random.randint(0,len(new_choose_city_1)-1)
                t_1 = new_choose_city_1[r_1]
                new_choose_city_1[r_1] = get_same_type_city(t_1,types)
                r_2 = random.randint(0, len(new_choose_city_2) - 1)
                t_2 = new_choose_city_2[r_2]
                new_choose_city_2[r_2] = get_same_type_city(t_2, types)
            l1 = gt(must_points,new_choose_city_1)
            tpn = l1.position
            sol_1, sumdis_1 = LKH(tpn,C)
            l1.fitness = sumdis_1
            zq.append(l1)
            l2 = gt(must_points,new_choose_city_2)
            tpn = l2.position
            sol_2, sumdis_2 = LKH(tpn,C)
            l2.fitness = sumdis_2
            zq.append(l2)
        """
        Individuals with low fitness were removed from the genetic samples after iteration, 
        and the total number of genetic samples was maintained at m
        """
        zq = sort_by_fitness(zq)
        zq= select_next_zq(zq,init_zq,m)
    """
    Find the individual with the highest fitness from the genetic sample 
    and draw the corresponding path
    """
    best_gt = zq[0]
    final_points = []
    for i in best_gt.position:
        final_points.append(points[i])
    tpn = best_gt.position
    sol, sumdis = LKH(tpn,C)
    best_tour = []
    for i in range(1, len(sol)):
        best_tour.append([sol[i - 1], sol[i]])
    points_r = turn_points(final_points, points)
    pt = turn_tour(best_tour, points_r)
    print(f"length of path = {sumdis}")
    print(f"path = {pt}")
    draw_points_type(np.array(points), types)
    plt_edges(best_tour, np.array(final_points), color='k')
    plt.title(sumdis)
    plt.show()

if __name__ == '__main__':
    """This file records the coordinate information and semantics of each point"""
    Coordinate_And_Semantics = "real_data.csv"

    """This file records the distance of the road network between any two points"""
    Distance_Of_Road_Network = "od_distance_metrix.csv"

    iterMax = 100       #Maximum number of iterations: 100

    """
    To get the value of time_limit, first pass the same files Coordinate_And_Semantics 
    and Distance_Of_Road_Network to Real_Data_SRC-LKH.
    Records the Real Data SRC-LKH run time.
    """
    time_limit = 0.3674011230468759

    GA_LKH_main(Coordinate_And_Semantics,Distance_Of_Road_Network,iterMax,time_limit)