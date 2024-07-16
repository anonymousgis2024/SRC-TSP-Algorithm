import copy
import math
import random
import time
import elkai
import numpy as np

from matplotlib import pyplot as plt
import scipy.spatial as ssp
from geopy.distance import geodesic
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

class lz:
    def __init__(self,M,O):
        self.M = M
        self.O = O
        self.position = self.M+self.O
        self.fitness = float('inf')
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_fitness = copy.deepcopy(self.fitness)
        self.pbest_O = O

def distance(points,i,j):
    if i!=j:
        dx = points[i][0]-points[j][0]
        dy = points[i][1]-points[j][1]
        dis = math.sqrt((dx*dx)+(dy*dy))
        return dis
    else:
        return float('inf')


def get_add_points(m,types):
    add_points = []
    while len(add_points)<m:
        a_p = []
        for i in range(len(types)):
            r = random.randint(0,len(types[i])-1)
            p = types[i][r]
            a_p.append(p)
        add_points.append(a_p)
    return add_points

def find_gbest(lzq):
    gbest = lzq[0]
    for l in lzq:
        if l.fitness<gbest.fitness:
            gbest = l
        else:
            continue
    return gbest

def get_same_type_city(t,types):
    for i in range(len(types)):
        if t in types[i]:
            r = random.randint(0,len(types[i])-1)
            return types[i][r]
    else:
        return -1

def Inertial_exploration(l,types):
    r = random.randint(0,len(l.O)-1)
    t = l.O[r]
    same_type_city = get_same_type_city(t,types)    #获得一个和城市t种类相同的城市
    if same_type_city!=-1:
        l.O[r] = same_type_city                     #将城市t替换掉
    l.position = l.M+l.O
    return l


def get_choose_city(list1,list2):
    list = []
    for i in range(len(list1)):
        a = random.randint(0, 1)
        if a==0:
            list.append(list1[i])
        elif a==1:
            list.append(list2[i])
    return list

def Personal_experience_exploration(l):
    new_choose_city = get_choose_city(l.O,l.pbest_O)
    l.O = new_choose_city
    l.position = l.M+l.O
    return l

def Social_exploration(l,gbest):
    new_choose_city = get_choose_city(l.O,gbest.O)
    l.O = new_choose_city
    l.position = l.M+l.O
    return l

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

def PSO_LKH_main(file1,file2,n,time_limit):
    """
    :param file1: This file records the coordinate information and semantics of each point
    :param file2: This file records the distance of the road network between any two points
    :param n: Maximum number of iterations
    :param time_limit: Upper limit for algorithm execution
    """

    m = 10  # The size of the particle swarm
    w = 1   # inertia factor
    c1 = 1  # Self-perception factor
    c2 = 1  # Social cognitive factor

    """
    By reading file 1, the location information and semantic attributes of the node are obtained
    """
    points,must_points,types = readDataFile_list(file1)
    np.random.seed(0)
    p_n = len(points)

    """
    By reading file2, the road network distance between any two points is obtained, 
    and the distance matrix is obtained
    """
    C = np.zeros((p_n, p_n))
    C = read_dis_mx(file2, C)
    """Generate initial population"""
    lzq = []
    add_points = get_add_points(m,types)    #Multiparticle formation
    for a in add_points:
        l = lz(must_points,a)               #Generate a particle
        test_points_num = l.position
        tpn = copy.deepcopy(test_points_num)
        test_points = []
        for i in test_points_num:
            test_points.append(points[i])
        sol, sumdis = LKH(tpn,C)  #The LKH algorithm is used to calculate the path corresponding to a certain particle
        l.fitness = l.pbest_fitness = sumdis
        lzq.append(l)

    gbest = copy.deepcopy(find_gbest(lzq))  #Find a global optimal solution from the initialized particle swarm, denoted gbest
    start_time = time.time()
    """Start iterative updates"""
    for i in range(n):
        """
        Each particle is iteratively updated according to the following formula
        V = wv+c1*random(0,1)*(pBest-X)+c2*random(0,1)*(gbest-X)
        """
        now = time.time()
        """If the algorithm runs longer than time_limit, it stops"""
        time_flag = now-start_time
        if time_flag>time_limit:
            print(f"Time spent = {time_flag}")
            break
        now_gbest = find_gbest(lzq)         #The optimal solution in the population is found after each iteration
        if now_gbest.fitness<gbest.fitness:
            gbest=copy.deepcopy(now_gbest)  #Update the global optimal solution

        for l in lzq:
            r = random.uniform(0,sum([w,c1,c2]))
            if r<w:
                l = Inertial_exploration(l,types)   #The particles are exploring freely
            elif r<w+c1:
                l = Personal_experience_exploration(l)  #The process of self-cognition: the current particle approaches the optimal solution it has found
            else:
                gbest_t = copy.deepcopy(gbest)          #Social cognitive processes: Approaching global optimal solutions for the entire population of current particles
                l = Social_exploration(l,gbest_t)

            test_points_num = l.position
            tpn = copy.deepcopy(test_points_num)
            test_points = []
            for i in test_points_num:
                test_points.append(points[i])
            """After each iteration, the fitness of each particle is recalculated"""
            sol, sumdis = LKH(tpn,C)
            l.fitness = sumdis

            """Compared to itself, if a particle finds its shortest path so far, 
            it updates the individual's optimal solution
            """
            if sumdis < l.pbest_fitness:
                l.pbest_position = l.position
                l.pbest_fitness = sumdis
                l.pbest_O = l.O
    test_points_num = gbest.position
    test_points = []
    for i in test_points_num:
        test_points.append(points[i])
    best_tour = []
    """Plot the path determined by the algorithm"""
    tpn = copy.deepcopy(test_points_num)
    sol, sumdis = LKH(tpn,C)
    for i in range(1, len(sol)):
        best_tour.append([sol[i - 1], sol[i]])
    draw_points_type(np.array(points), types)
    plt_edges(best_tour, np.array(test_points), color='k')
    plt.title(sumdis)
    plt.show()
    points_r = turn_points(test_points, points)
    pt = turn_tour(best_tour, points_r)
    print(f"length of path = {sumdis}")
    print(f"path = {pt}")


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

    PSO_LKH_main(Coordinate_And_Semantics,Distance_Of_Road_Network,iterMax,time_limit)


