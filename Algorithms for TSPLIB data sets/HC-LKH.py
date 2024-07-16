import copy
import math
import os
import random
import time
import elkai
import numpy as np
import scipy.spatial as ssp
from matplotlib import pyplot as plt
def readDataFile_list(fileName):
    dataList = list()
    with open(r"../dataset/TSPLIB/"+fileName,'r') as fr:
        fr.readline()#跳过标题行
        for line in fr:
            strs = line.strip().split(",")
            dataList.append((float(strs[1]),float(strs[2])))
    return dataList

def draw_points_type(points,types):
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
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
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    #绘制连线
    for p1,p2 in edges_touple:
        plt.plot([points[p1][0],points[p2][0]], [points[p1][1],points[p2][1]], color=color, linewidth=1)

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
    #使用了欧几里得公式算出距离C
    if i!=j:
        dx = points[i][0]-points[j][0]
        dy = points[i][1]-points[j][1]
        dis = math.sqrt((dx*dx)+(dy*dy))
        return dis
    else:
        return float('inf')

#计算C矩阵
def count_C_m(points,p_n):
    #将距离记录在矩阵C中
    C = []
    for i in range(p_n):
        c_element = []
        for j in range(p_n):
            # if i==j:
            #     c_element.append()
            a = distance(points,i,j)
            # print(i,j,a)
            c_element.append(round(a,2))
        C.append(c_element)
    # print(C)
    return C

def count_len(tour,C):
    l = 0
    for edge in tour:
        f = edge[0]
        t = edge[1]
        add = C[f][t]
        l +=add
    return l

def types_m_init(n, left_point_num):
    types = []
    ave = int(len(left_point_num)/n)
    t = copy.deepcopy(left_point_num)
    for i in range(n):
        if i == n-1:
            types.append(t)
        else:
            random_sample = list(np.random.choice(t, ave, replace=False))
            types.append(random_sample)
            t = list(set(t)-set(random_sample))
    return types

def get_add_points(m,types):
    """
    :param m:       粒子群规模
    :param types:   划分存储的可选城市
    :return:
    """
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

def LKH(city_condition):
    # 计算距离矩阵，此处用坐标直接计算最短距离，实际交通规划中可能会有现成的最短距离矩阵
    def genDistanceMat(x, y):
        X = np.array([x, y])
        distMat = ssp.distance.pdist(X.T)
        distMat = ssp.distance.squareform(distMat)
        return distMat

    x, y = city_condition[:, 0], city_condition[:, 1]
    distance = genDistanceMat(x, y)

    # 计算所得方案的线路总长度
    def cal_fitness(sol):
        tot_len = np.sum(distance[sol[:-1], sol[1:len(sol)]])
        return tot_len

    sol = elkai.solve_int_matrix(distance)
    # sol = elkai.solve_float_matrix(distance,runs=10)#允许浮点距离
    sol.append(0)
    '''这个函数计算出的是有回路的TSP问题，但返回的解方案sol没有给出完整解方案(少一个终点0),故我们在代码中在解方案最末尾加上了编号0'''

    # print("最优解方案:", sol)
    sumdis = cal_fitness(sol)
    # print("最优解总长度:", sumdis)
    return sol, sumdis

def get_fitness(init_solve,must_points,points):
    t = init_solve+must_points
    test_points = []
    for p in t:
        test_points.append(points[p])
    sol, sumdis = LKH(np.array(test_points))
    return sumdis



def HC_LKH_main(fileName, semantics, must_per, iterMax, total, timeMax):
    """
    :param fileName:        Name of the data set
    :param semantics:       Semantic category
    :param must_per:        |M|=must_per*(Number of nodes in data set)
    :param iterMax:         The maximum number of iterations
    :param total:           The target length of the path
    :param timeMax:         Maximum running time of the algorithm
    """
    points = readDataFile_list(fileName)    #Read data set
    p_n = len(points)
    point_num = [i for i in range(p_n)]
    must_p_n = int(must_per * p_n)
    np.random.seed(0)
    must_points = list(np.random.choice(point_num, must_p_n, replace=False))    #Structural set M
    left_points = list(set(point_num) - set(must_points))                       #Remaining node
    types = types_m_init(semantics, left_points)    #The set S is divided into n classes according to semantics
    init_solve = get_add_points(1, types)           #Determine an initial solution
    best_fitness = get_fitness(init_solve,must_points,points)

    solve = copy.deepcopy(init_solve)
    start_time = time.time()
    #Start iterative updates
    for i in range(iterMax):
        neighbourhood = get_neighbourhood(solve,types)  #The neighborhood of the initial solution is obtained and changes in the field
        add_points = disturbance(solve,neighbourhood)
        add_points.append(solve)
        for p in add_points:
            fitness = get_fitness(p,must_points,points)
            if fitness<best_fitness:
                init_solve = p
                best_fitness = fitness
                break
        else:
            r = random.randint(0,len(init_solve)-1)
            r2 = random.randint(0,len(types[r])-1)
            solve = copy.deepcopy(init_solve)
            solve[r] = types[r][r2]
        now = time.time()
        """If a better path than the target path is found, or the algorithm runs longer than online, it stops"""
        if best_fitness<total+1 or now-start_time>=timeMax:
            break

    end_time = time.time()
    time_cost = end_time - start_time   #Calculate the running time of the algorithm.
    print(f"Time spent:{time_cost}s")
    #Plot the path determined by the algorithm
    test_points_num = init_solve+must_points
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
    This file records the coordinate information of each point.
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
    HC_LKH_main(fileName, semantics, percent, iterations, total, timeMax)



