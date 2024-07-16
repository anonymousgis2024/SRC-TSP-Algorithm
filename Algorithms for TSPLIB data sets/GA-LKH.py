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


def readDataFile_list(fileName):
    dataList = list()
    with open(r"../dataset/TSPLIB/"+fileName,'r') as fr:
        fr.readline()
        for line in fr:
            strs = line.strip().split(",")
            dataList.append((float(strs[1]),float(strs[2])))
    return dataList

def types_m_init(n, left_point_num):
    types = []
    ave = int(len(left_point_num)/n)
    t = copy.deepcopy(left_point_num)
    for i in range(n):
        if i == n-1:
            types.append(t)
        else:
            random_sample = list(np.random.choice(t,ave,replace = False))
            types.append(random_sample)
            t = list(set(t)-set(random_sample))
    return types

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

def distance(points,i,j):
    if i!=j:
        dx = points[i][0]-points[j][0]
        dy = points[i][1]-points[j][1]
        dis = math.sqrt((dx*dx)+(dy*dy))
        return dis
    else:
        return float('inf')

def get_dis_mat(points,p_n):
    dis_mat = []
    for i in range(p_n):
        dis_list = []
        for j in range(p_n):
            d = distance(points,i,j)
            dis_list.append(round(d,2))
        dis_mat.append(dis_list)
    return dis_mat

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

def LKH(city_condition):
    def genDistanceMat(x, y):
        X = np.array([x, y])
        distMat = ssp.distance.pdist(X.T)
        distMat = ssp.distance.squareform(distMat)
        return distMat
    x, y = city_condition[:, 0], city_condition[:, 1]
    distance = genDistanceMat(x, y)
    def cal_fitness(sol):
        tot_len = np.sum(distance[sol[:-1], sol[1:len(sol)]])
        return tot_len

    sol = elkai.solve_int_matrix(distance)
    # sol = elkai.solve_float_matrix(distance,runs=10)
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
        dx = points[i][0]-points[j][0]
        dy = points[i][1]-points[j][1]
        dis = math.sqrt((dx*dx)+(dy*dy))
        return dis
    else:
        return float('inf')


def count_C_m(points,p_n):
    C = []
    for i in range(p_n):
        c_element = []
        for j in range(p_n):
            a = distance(points,i,j)
            c_element.append(round(a,2))
        C.append(c_element)
    return C

def count_len(tour,C):
    l = 0
    for edge in tour:
        f = edge[0]
        t = edge[1]
        add = C[f][t]
        l +=add
    return l

def GA_LKH_main(fileName, n, iterations, must_per, total_fitness, timeMax):
    """
    :param fileName:        Name of the data set
    :param n:               Semantic category
    :param iterations:      The maximum number of iterations
    :param must_per:        |M|=must_per*(Number of nodes in data set)
    :param total_fitness:   The target length of the path
    :param timeMax:         Maximum running time of the algorithm
    """
    m = 25   # The size of the genetic sample
    v = 0.2  # mutation factor
    points = readDataFile_list(fileName)    #Read data set
    p_n = len(points)
    point_num = [i for i in range(p_n)]
    must_p_n = int(must_per*p_n)
    np.random.seed(0)
    must_points = list(np.random.choice(point_num, must_p_n, replace=False))    #Structural set M
    left_points= list(set(point_num)-set(must_points))                          #Remaining node
    types = types_m_init(n, left_points)       #The set S is divided into n classes according to semantics


    #n nodes with different semantics are selected to form a genetic sample
    zq = []
    add_points = get_add_points(m,types)    #m genetic samples are generated
    for a in add_points:
        l = gt(must_points,a)
        test_points = []
        for p in l.position:
            test_points.append(points[p])
        sol, sumdis = LKH(np.array(test_points))
        l.fitness = sumdis
        zq.append(l)
    fitness_list = get_fitness(zq)
    init_zq = copy.deepcopy(zq)

    start_time = time.time()
    for i in range(iterations):         #genetic evolution
        for j in range(m):              #Select two genetic samples
            p1, p2 = selection(fitness_list, 2)
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

            #Start mutating operation
            if random.random() < v:
                r_1 = random.randint(0,len(new_choose_city_1)-1)
                t_1 = new_choose_city_1[r_1]
                new_choose_city_1[r_1] = get_same_type_city(t_1,types)
                r_2 = random.randint(0, len(new_choose_city_2) - 1)
                t_2 = new_choose_city_2[r_2]
                new_choose_city_2[r_2] = get_same_type_city(t_2, types)
            l1 = gt(must_points,new_choose_city_1)

            test_points = []
            for p in l1.position:
                test_points.append(points[p])
            sol, sumdis = LKH(np.array(test_points))
            l1.fitness = sumdis
            zq.append(l1)
            l2 = gt(must_points,new_choose_city_2)

            test_points = []
            for p in l2.position:
                test_points.append(points[p])
            sol, sumdis = LKH(np.array(test_points))
            l2.fitness = sumdis
            zq.append(l2)
        zq = sort_by_fitness(zq)
        zq= select_next_zq(zq,init_zq,m)
        t = time.time()
        """If a better path than the target path is found, or the algorithm runs longer than online, it stops"""
        if zq[0].fitness<=total_fitness or t-start_time>=timeMax:
            break
    end_time = time.time()
    #Calculate the running time of the algorithm.
    time_cost = end_time - start_time
    print(f"Time spent:{time_cost}s")

    #Plot the path determined by the algorithm
    test_points_num = zq[0].position
    test_points = []
    for i in test_points_num:
        test_points.append(points[i])
    best_tour = []
    sol, sumdis = LKH(np.array(test_points))
    for i in range(1, len(sol)):
        best_tour.append([sol[i - 1], sol[i]])
    """
    Draw the final path of the algorithm
    """
    draw_points_type(np.array(points), types)
    plt_edges(best_tour, np.array(test_points), color='k')
    plt.title(sumdis)
    plt.show()
    points_r = turn_points(test_points, points)
    pt = turn_tour(best_tour, points_r)
    print(f"length of path = {sumdis}")
    print(f"path = {pt}")


if __name__ == '__main__':
    """
    You can replace fileName with any of the data sets in the following list.
    Specifically, you can select any dataset in the dataset/TSPLIB folder
    ["pr76.csv","kroA100.csv","kroC100.csv","pr124.csv","pr136.csv","ch150.csv","kroA150.csv"]
    This file records the coordinate information of each point
    """
    fileName = "kroA150.csv"

    iterations = 100    #Maximum number of iterations: 100


    semantics = 5       #There are 5 semantics in the data set
    """
    You can replace percent with any of the values in the list below.
    [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    """
    percent = 0.4       #|M|=percent*(Number of nodes in data set fileName).M stands for mandatory node set.

    timeMax = 7200      #Maximum running time of the algorithm.Once the running time of the algorithm exceeds this value, the algorithm will stop after the current iteration

    """
    To get the total value, first set the same three parameters fileName, percent, semantics for SRC-LKH.
    Finally use the path length obtained by SRC-LKH as the total value
    """
    total = 17918  # When the path of the algorithm is less than or equal to the total value, the algorithm stops
    GA_LKH_main(fileName, semantics, iterations, percent, total, timeMax)