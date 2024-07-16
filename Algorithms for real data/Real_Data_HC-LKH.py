import copy
import math
import random
import time
import elkai
import numpy as np
from geopy.distance import geodesic
from matplotlib import pyplot as plt
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
            c_element.append(round(distance(points,i,j),2))
        C.append(c_element)
    return C

def get_add_points(m,types):
    add_points = []
    while len(add_points)<m:
        a_p = []
        for i in range(len(types)):
            r = random.randint(0,len(types[i])-1)
            p = types[i][r]
            a_p.append(p)
        add_points.append(a_p)
    return add_points[0]

def get_neighbourhood(init_solve,types):
    neighbourhood = []
    for i in range(len(init_solve)):
        n = []
        t = init_solve[i]
        index = types[i].index(t)
        a = [index-2,index-1,index+1,index+2]
        for j in a:
            if j>=0 and j<len(types[i]):
                n.append(types[i][j])
        neighbourhood.append(n)
    return neighbourhood

def disturbance(init_solve,neighbourhood):
    add_points = []
    for i in range(len(init_solve)):
        init = copy.deepcopy(init_solve)
        for j in range(len(neighbourhood[i])):
            init[i] = neighbourhood[i][j]
            add_points.append(init)
    return add_points

def LKH(city_condition,city_num,od_dis_mx):
    distance = np.zeros((len(city_num),len(city_num)))
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

def read_dis_mx(file,dis_mx):
    with open(r"../dataset/real_data/"+file,'r') as fr:
        fr.readline()
        for line in fr:
            line = line.strip()
            atrs = line.split(",")
            o1,d1,dis = int(atrs[0]),int(atrs[1]),float(atrs[2])
            dis_mx[o1-1][d1-1]=dis
    return dis_mx

def get_fitness(init_solve,must_points,points,p_n,C):
    tpn = init_solve+must_points
    test_points = []
    for p in tpn:
        test_points.append(points[p])

    LKH(np.array(test_points), tpn, C)
    sol, sumdis = LKH(np.array(test_points),tpn,C)
    return sol,sumdis

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

def HC_LKH_main(file1,file2,iterMax,time_limit):
    """
    :param file1: This file records the coordinate information and semantics of each point
    :param file2: This file records the distance of the road network between any two points
    :param iterMax: Maximum number of iterations
    :param time_limit: Upper limit for algorithm execution
    """

    """
    By reading file 1, the location information and semantic attributes of the node are obtained
    """
    points,must_points,types = readDataFile_list(file1)
    p_n = len(points)
    np.random.seed(0)
    init_solve = get_add_points(1, types)  #Determine an initial solution

    """
    By reading file2, the road network distance between any two points is obtained, 
    and the distance matrix is obtained
    """
    C = np.zeros((p_n, p_n))
    C = read_dis_mx(file2, C)
    bs,best_fitness = get_fitness(init_solve,must_points,points,p_n,C)
    solve = copy.deepcopy(init_solve)

    """
    genetic evolution
    The algorithm is not allowed to iterate more than n times.
    """
    start_time = time.time()
    for i in range(iterMax):
        now = time.time()
        time_flag = now-start_time
        """If the algorithm runs longer than time_limit, it stops"""
        if time_flag>time_limit:
            print(f"Time spent = {time_flag}s")
            break
        neighbourhood = get_neighbourhood(solve,types)  #The neighborhood of the initial solution is obtained and changes in the field
        add_points = disturbance(solve,neighbourhood)
        add_points.append(solve)
        for p in add_points:
            s,fitness = get_fitness(p,must_points,points,p_n,C)
            if fitness<best_fitness:
                init_solve = p
                best_fitness = fitness
                bs = s
                break
        else:
            r = random.randint(0,len(init_solve)-1)
            r2 = random.randint(0,len(types[r])-1)
            solve = copy.deepcopy(init_solve)
            solve[r] = types[r][r2]

        # Plot the path determined by the algorithm
        best_tour = []
        for i in range(1, len(bs)):
            best_tour.append([bs[i - 1], bs[i]])
        final_points_num = init_solve + must_points
        final_points = []
        for p in final_points_num:
            final_points.append(points[p])
        points_r = turn_points(final_points, points)
        pt = turn_tour(best_tour, points_r)
        my_pt = count_len(pt, C)
        print(f"length of path = {my_pt}")
        print(f"path = {pt}")
        draw_points_type(np.array(points), types)
        plt_edges(best_tour, np.array(final_points), color='k')
        plt.title(my_pt)
        plt.show()


if __name__ == '__main__':
    """This file records the coordinate information and semantics of each point"""
    Coordinate_And_Semantics = "real_data.csv"


    """This file records the distance of the road network between any two points"""
    Distance_Of_Road_Network = "od_distance_metrix.csv"

    iterMax = 100   #Maximum number of iterations: 100
    """
    To get the value of time_limit, first pass the same files Coordinate_And_Semantics 
    and Distance_Of_Road_Network to Real_Data_SRC-LKH.
    Records the Real Data SRC-LKH run time.
    """
    time_limit = 0.3674011230468759
    HC_LKH_main(Coordinate_And_Semantics,Distance_Of_Road_Network,iterMax,time_limit)

