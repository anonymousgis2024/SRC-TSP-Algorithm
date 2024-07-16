import itertools
import math
import copy
import os
import random
import time
from itertools import combinations
import elkai
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
import scipy.spatial as ssp
from geopy.distance import geodesic

#画点
def plt_points(points):
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    for i,(x,y) in enumerate(points):
        plt.scatter(x, y, color='r', s=10)
        plt.text(x, y, str(i), fontsize=6, color='b')

def draw_points_type(points,types):
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    n = len(types)
    for i, (x, y) in enumerate(points):
        if i in types[0]:
            plt.scatter(x,y,color = 'b',s = 10,marker='*')
            plt.text(x, y, str(i), fontsize=6, color='b')
        elif i in types[1]:
            plt.scatter(x, y, color='g', s=10,marker='x')
            plt.text(x, y, str(i), fontsize=6, color='b')
        elif i in types[2]:
            plt.scatter(x, y, color='y', s=10,marker='s')
            plt.text(x, y, str(i), fontsize=6, color='b')
        elif i in types[3]:
            plt.scatter(x, y, color='k', s=10,marker='p')
            plt.text(x, y, str(i), fontsize=6, color='b')
        elif i in types[4]:
            plt.scatter(x, y, color='r', s=10)
            plt.text(x, y, str(i), fontsize=6, color='b')
        elif i in types[5]:
            plt.scatter(x, y, color='b', s=10)
            plt.text(x, y, str(i), fontsize=6, color='b')
        # plt.scatter(x, y, color='r', s=10)
        else:
            plt.scatter(x, y, color='k', s=3)

#画边
def plt_edges(edges_touple, points, color):
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    #绘制连线
    for p1,p2 in edges_touple:
        plt.plot([points[p1][0],points[p2][0]], [points[p1][1],points[p2][1]], color=color, linewidth=1)

#读取数据集
def readDataFile_list(fileName):
    dataList = list()
    with open(r"../dataset/TSPLIB/"+fileName,'r') as fr:
        fr.readline()#跳过标题行
        for line in fr:
            strs = line.strip().split(",")
            dataList.append((float(strs[1]),float(strs[2])))
    return dataList

#计算点之间的距离
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
            a = distance(points,i,j)
            c_element.append(round(a,2))
        C.append(c_element)
    return C

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

def get_point_type(types,p):
    for i in range(len(types)):
        if p in types[i]:
            return i

def get_stop_gen_flag(types):
    for type in types:
        if len(type)>0:
            return 0
    else:
        return 1

def get_unable_flag(p,free_types):
    #如果p是可以加入的点，则返回0，否则返回1
    for type in free_types:
        if p in type:
            return 0
    else:
        return 1

def edit_free_types(types,free_types,gen_point):
    for p in gen_point:
        p_type = get_point_type(types,p)
        free_types[p_type].remove(p)
    return free_types

def get_gen_points(neighbors,types):
    free_types = copy.deepcopy(types)
    gen_points = []
    stop_gen_flag = 0
    while stop_gen_flag != 1:
        gen_point = []
        gen_point_type = []
        i = random.randint(0,len(free_types)-1)
        if len(free_types[i])>0:
            random.shuffle(free_types[i])
            gen_point.append(free_types[i][0])
            gen_point_type.append(i)
        else:
            continue
        start_flag = 0
        not_fit_list = []
        while start_flag<len(gen_point):
            n = neighbors[gen_point[start_flag]]
            for p in n:
                #找到了某个点的全部邻接点，看看能否构成一个簇
                unable_flag = get_unable_flag(p,free_types)
                if unable_flag!=1:
                    p_type = get_point_type(types,p)
                    if (p not in gen_point) and (p_type not in gen_point_type) and (p not in not_fit_list):
                        gen_point.append(p)
                        gen_point_type.append(p_type)
                    elif (p not in gen_point) and (p_type in gen_point_type):
                        not_fit = neighbors[p]
                        for not_fit_point in not_fit:
                            if not_fit_point not in not_fit_list:
                                not_fit_list.append(not_fit_point)
                else:
                    continue
            start_flag+=1
        free_types = edit_free_types(types,free_types,gen_point)
        gen_points.append(gen_point)
        stop_gen_flag = get_stop_gen_flag(free_types)
    return gen_points

def sort_by_dis(neighbors,C):
    for i in range(len(neighbors)):
        if len(neighbors[i])>0:
            dis_list = []
            for j in neighbors[i]:
                dis = C[i][j]
                dis_list.append(dis)
            neighbors[i] = [x for _,x in sorted(zip(dis_list,neighbors[i]))]
    return neighbors

def generate_subsets(s):
    subsets = []
    n = len(s)
    for r in range(n+1):
        for subset in combinations(s,r):
            a = list(subset)
            if len(a)>0:
                subsets.append(a)
    return subsets

def get_gen_point_type(gen_point,types):
    gen_point_type = []
    for p in gen_point:
        p_type = get_point_type(types,p)
        gen_point_type.append(p_type)
    gen_point_type.sort()
    return gen_point_type

def get_fit_points(type,gen_point,types,neighbors):
    points = []
    t = []
    for p in gen_point:
        t1 = get_point_type(types,p)
        t.append(t1)
    for i in type:
        index = t.index(i)
        points.append(gen_point[index])
    #需要一段代码来判断这些找出来的点是否相邻
    fit_points = []
    p = points[0]
    test_points = [p]
    start_flag = 0
    while start_flag<len(test_points):
        n_points = neighbors[test_points[start_flag]]
        for n_p in n_points:
            if (n_p in points) and (n_p not in test_points):
                test_points.append(n_p)
        start_flag+=1
    l = list(set(points)-set(test_points))
    if len(l)==0:
        fit_points = points
    return fit_points

def create_paruto_list(subsets,paruto_list,gen_points,types,neighbors):
    #已知一个簇，将这个簇分解为
    """
    :param subsets:     这是一个列表，其中记录了集合[[0],[1],``````[1,2,3,4,5]]
    :param paruto_list: 这是一个列表，用于记录下前沿，与上一个参数对应，[1]这个类对应下标1，paruto[1]中就记录1这个type种最优秀的点
    :param gen_points:  所有的簇
    :param types:       所有点的种类
    :return:            将paruto_list返回
    """
    for gen_point in gen_points:
        gen_point_type = get_gen_point_type(gen_point,types)
        gen_point_types = generate_subsets(set(gen_point_type))
        #取出一个簇，计算这个簇的种类，并计算这个簇种类的子集————计算这个簇可以拆分成什么种类的子集
        for type in gen_point_types:
            #取出其中一个种类，找对应的点
            points = get_fit_points(type,gen_point,types,neighbors)
            index = subsets.index(type)
            if (len(points)>0) and (points not in paruto_list[index]):
                paruto_list[index].append(points)
    return paruto_list

def create_paruto_list2(subsets,paruto_list,gen_points,types):
    for gen_point in gen_points:
        gen_point_type = get_gen_point_type(gen_point,types)
        index = subsets.index(gen_point_type)
        paruto_list[index].append(gen_point)
    return paruto_list

def get_near_point_by_point(f_point,C,must_points):
    dis_list = []
    for m_p in must_points:
        dis_list.append(C[f_point][m_p])
    special_p = [x for _, x in sorted(zip(dis_list,must_points))]
    return special_p[:2]

def get_near_point_by_point_2(f_point,C,must_points):
    dis_list = []
    for m_p in must_points:
        dis_list.append(C[f_point][m_p])
    # dis_list = copy.deepcopy(C[f_point][:must_p_n])
    # p = []
    # for i in range(must_p_n):
    #     p.append(i)
    special_p = [x for _, x in sorted(zip(dis_list,must_points))]
    return special_p[:3]

def get_dis_to_tour(gen_point,C,must_points,points):
    """
    :param gen_point:           这是一个或几个点构成的簇，list类型
    :param C:                   距离矩阵
    :return:                    返回的是簇到初始路径的距离
    """
    p = gen_point[0]        #一个点作为一个簇的情况
    d = []
    dis = 0
    special_points = get_near_point_by_point_2(p,C,must_points)
    d_0 = C[p][special_points[0]]
    d_1 = C[p][special_points[1]]
    d_2 = C[p][special_points[2]]
    d_01 = d_0+d_1-C[special_points[0]][special_points[1]]
    d_02 = d_0+d_2-C[special_points[0]][special_points[2]]
    d_12 = d_1+d_2-C[special_points[1]][special_points[2]]
    d_list = [d_0,d_1,d_01,d_2,d_12]
    d_list.sort()
    for i in d_list:
        if i>=0:
            return i
    return dis

def get_dis_to_tour_2(gen_point,C,must_points,points):
    """
    :param gen_point:           这是一个或几个点构成的簇，list类型
    :param C:                   距离矩阵
    :return:                    返回的是簇到初始路径的距离
    """
    p = gen_point[0]        #一个点作为一个簇的情况
    d = []
    dis = 0
    special_points = get_near_point_by_point_2(p,C,must_points)
    d_0 = C[p][special_points[0]]
    d_1 = C[p][special_points[1]]
    d_2 = C[p][special_points[2]]
    d_list = [d_0,d_1,d_2]
    d_list.sort()
    for i in d_list:
        if i>=0:
            return i
    return dis

def get_dis_p(p,tree_p,C):
    dis = []
    for t_p in tree_p:
        dis.append(C[p][t_p])
    min_dis = min(dis)
    index = dis.index(min_dis)
    return min(dis)

def get_minmal_tree_dis(gen_point,C,must_points):
    tree_p = []
    left_p = copy.deepcopy(gen_point)
    dis = 0
    while len(left_p)>0:
        if len(tree_p)==0:
            tree_p.append(gen_point[0])
            left_p.remove(gen_point[0])
        else:
            min_dis = -1
            min_dis_p = -1
            for p in left_p:
                p_dis = get_dis_p(p,tree_p,C)
                if min_dis == -1:
                    min_dis = p_dis
                    min_dis_p = p
                else:
                    if p_dis<min_dis:
                        min_dis = p_dis
                        min_dis_p = p
            tree_p.append(min_dis_p)
            left_p.remove(min_dis_p)
            dis+=min_dis
            # print(f"dis={dis},tree_p={tree_p}")
    del_edges = []
    for p in gen_point:
        l1 = []
        for m_p in must_points:
            l1.append(C[p][m_p])
        f = min(l1)
        f_p = l1.index(f)
        special_point = get_near_point_by_point(f_p,C,must_points)
        del_edges.append(special_point)
    dis_tree_tour = []
    for e in del_edges:
        p1 = e[0]
        p2 = e[1]
        p_p1 = []
        p_p2 = []
        for p in gen_point:
            p_p1.append(C[p][p1])
            p_p2.append(C[p][p2])
        dis_tree_tour.append((sum(p_p1)/len(p_p1))+(sum(p_p2)/len(p_p2))-C[p1][p2])
    dis += min(dis_tree_tour)*0.3+sum(dis_tree_tour)/len(dis_tree_tour)*0.7
    return dis

def get_leading_edge(paruto_list,C,must_points,points):
    paruto_leading_edge = []
    paruto_leading_dis = []
    for i in range(len(paruto_list)):
        paruto_leading_edge.append([])
        paruto_leading_dis.append([])
        if len(paruto_list[i])>0:
            element_test_len = len(paruto_list[i][0])
            if element_test_len==1:
                for gen_point in paruto_list[i]:
                    gen_point_dis_to_tour = get_dis_to_tour(gen_point,C,must_points,points)
                    paruto_leading_edge[i].append(gen_point)
                    paruto_leading_dis[i].append(gen_point_dis_to_tour)
                paruto_leading_edge[i] = [x for _, x in sorted(zip(paruto_leading_dis[i], paruto_leading_edge[i]))]
                paruto_leading_edge[i] = paruto_leading_edge[i][:10]
                paruto_leading_dis[i].sort()
                paruto_leading_dis[i] = paruto_leading_dis[i][:10]
            else:
                for gen_point in paruto_list[i]:
                    gen_point_dis_to_tour = get_minmal_tree_dis(gen_point,C,must_points)
                    paruto_leading_edge[i].append(gen_point)
                    paruto_leading_dis[i].append(gen_point_dis_to_tour)
                paruto_leading_edge[i] = [x for _, x in sorted(zip(paruto_leading_dis[i], paruto_leading_edge[i]))]
                paruto_leading_edge[i] = paruto_leading_edge[i][:10]
                paruto_leading_dis[i].sort()
                paruto_leading_dis[i] = paruto_leading_dis[i][:10]
    return paruto_leading_edge,paruto_leading_dis

def get_leading_edge_2(paruto_list,C,must_points,points):
    paruto_leading_edge = []
    paruto_leading_dis = []
    for i in range(len(paruto_list)):
        paruto_leading_edge.append([])
        paruto_leading_dis.append([])
        if len(paruto_list[i])>0:
            element_test_len = len(paruto_list[i][0])
            if element_test_len==1:
                for gen_point in paruto_list[i]:
                    # print(gen_point)
                    #需要计算这个簇到路径的距离
                    gen_point_dis_to_tour = get_dis_to_tour_2(gen_point,C,must_points,points)
                    paruto_leading_edge[i].append(gen_point)
                    paruto_leading_dis[i].append(gen_point_dis_to_tour)
                paruto_leading_edge[i] = [x for _, x in sorted(zip(paruto_leading_dis[i], paruto_leading_edge[i]))]
                paruto_leading_edge[i] = paruto_leading_edge[i][:10]
                paruto_leading_dis[i].sort()
                paruto_leading_dis[i] = paruto_leading_dis[i][:10]
            else:
                for gen_point in paruto_list[i]:
                    #gen_point中含有多个点时，该如何计算这个簇到路径的距离
                    #将gen_point生成一个最小生成树，加上gen_point中的点到路路径的最短距离
                    gen_point_dis_to_tour = get_minmal_tree_dis(gen_point,C,must_points)
                    paruto_leading_edge[i].append(gen_point)
                    paruto_leading_dis[i].append(gen_point_dis_to_tour)
                paruto_leading_edge[i] = [x for _, x in sorted(zip(paruto_leading_dis[i], paruto_leading_edge[i]))]
                paruto_leading_edge[i] = paruto_leading_edge[i][:10]
                paruto_leading_dis[i].sort()
                paruto_leading_dis[i] = paruto_leading_dis[i][:10]

    # for j in range(len(paruto_leading_edge)):
    #     print(paruto_leading_edge[j],paruto_leading_dis[j])
    return paruto_leading_edge,paruto_leading_dis

def calculate_combinations(lst, n):
    combinations = list(itertools.combinations(lst, n))
    combinations = [list(comb) for comb in combinations]
    return combinations

def split_list(original_list,i):
    result = []

    if i==1:
        # return original_list
        result.append([original_list])
    elif i==len(original_list):
        e = []
        for j in original_list:
            e.append([j])
        # return e
        result.append(e)
    else:
        n = len(original_list)
        max_choose = n-i+1
        for j in range(1,max_choose+1):
            part_list = calculate_combinations(original_list,j)
            for part in part_list:
                left_list = list(set(original_list)-set(part))
                r = split_list(left_list,i-1)
                for k in r:
                    # print(f"k={k}")
                    # print([part]+k)
                    e = k[0]
                    if type(e)==type([]):
                        test = [part]+k
                    elif type(e)==type(0):
                        test = [part]+[k]
                    sort_test = sorted(test)
                    if sort_test not in result:
                        result.append(sort_test)

    return result

def find_add_method(subsets,paruto_leading_edge,paruto_leading_dis,types):

    need_types = subsets[-1]

    methods = []
    for i in range(1, len(need_types) + 1):
        results = split_list(need_types, i)
        for j in results:
           methods.append(j)
    add_points = []
    add_dis = []
    add_method = []
    for method in methods:
        a_p = []
        dis = 0
        for p in method:
            index = subsets.index(p)
            if len(paruto_leading_edge[index])>0:
                for i in paruto_leading_edge[index][0]:
                    a_p.append(i)
                dis += paruto_leading_dis[index][0]
            else:
                break
        if len(a_p) == len(need_types):
            add_method.append(method)
            add_points.append(a_p)
            add_dis.append(dis)
    add_p = [x for _, x in sorted(zip(add_dis,add_points))]
    add_m = [x for _, x in sorted(zip(add_dis, add_method))]
    return add_m,add_p

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

def count_len(tour,C):
    l = 0
    for edge in tour:
        f = edge[0]
        t = edge[1]
        add = C[f][t]
        l +=add
    return l

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
        new_e = tuple([points_r[e[0]],points_r[e[1]]])
        pt.append(new_e)
    return pt

def  SRC_LKH_main(fileName,must_per,n):
    """
    :param fileName:    Name of the data set
    :param must_per:    |M|=must_per*(Number of nodes in data set)
    :param n:           Semantic category
    """

    points = readDataFile_list(fileName)    #Read data set
    neighbors = []
    for i in range(len(points)):
        neighbors.append([])
    vor = Voronoi(np.array(points))         #All nodes in the data set are analyzed using the Tyson polygon.
    p_n = len(points)
    C = count_C_m(points,p_n)               #The distance matrix is calculated using the horizontal and vertical coordinates of the nodes

    point_num = [i for i in range(p_n)]
    must_p_n = int(must_per*p_n)
    np.random.seed(0)
    must_points = list(np.random.choice(point_num, must_p_n, replace=False))    #Structural set M
    left_points= list(set(point_num)-set(must_points))                          #Remaining node
    types = types_m_init(n, left_points)            #The set S is divided into n classes according to semantics

    """
    Using the Tyson polygon to judge whether any two semantic nodes are adjacent, 
    the adjacency relation table is further obtained.
    """
    for c1, c2 in vor.ridge_points:
        if (c1 not in must_points) and (c2 not in must_points):
            neighbors[c1].append(c2)
            neighbors[c2].append(c1)
    neighbors = sort_by_dis(neighbors,C)

    """
    Clustering of adjacent and semantically different nodes into clusters.
    """
    gen_points = get_gen_points(neighbors,types)

    """
    Analyze the obtained clusters and further extract high-quality subsets.
    """
    s = set()
    for i in range(len(types)):
        s.add(i)
    subsets = generate_subsets(s)       #The results of semantic constraint partitioning are discussed
    paruto_list = []
    for i in range(len(subsets)):
        paruto_list.append([])
    """
    The subset of each cluster is computed and stored
    """
    paruto_list = create_paruto_list(subsets,paruto_list,gen_points,types,neighbors)

    """
    Compute the cost of a subset of length one
    """
    paruto_leading_edge,paruto_leading_dis = get_leading_edge(paruto_list,C,must_points,points)

    """
    Calculate the cost of subsets of length greater than one
    """
    paruto_leading_edge_2,paruto_leading_dis_2 = get_leading_edge_2(paruto_list,C,must_points,points)

    """
    The corresponding subset is found according to the semantic combination mode, 
    and the cost of the subset is added as the cost of the corresponding combination mode
    """
    add_methods,add_points = find_add_method(subsets,paruto_leading_edge,paruto_leading_dis,types)

    best_tour = []
    test_points = []
    for j in must_points:
        test_points.append(points[j])
    M = copy.deepcopy(test_points)

    """
    Select the combination with the least cost to form the initial path.
    """
    for k in add_points[0]:
        test_points.append(points[k])
    sol, sumdis = LKH(np.array(test_points))
    for i in range(1, len(sol)):
        best_tour.append([sol[i - 1], sol[i]])
    best_fitness = copy.deepcopy(sumdis)
    best_add = copy.deepcopy(add_points[0])
    best_points = copy.deepcopy(test_points)

    """
    Construct a set of candidates for perturbation.
    """
    candidates = []
    for x in range(n):
        candidate = []
        if len(paruto_leading_edge_2[x])>=2:
            candidate.append(paruto_leading_edge_2[x][0][0])
            candidate.append(paruto_leading_edge_2[x][1][0])
        elif len(paruto_leading_edge_2[x])==1:
            candidate.append(paruto_leading_edge_2[x][0][0])
        if len(paruto_leading_edge[x])>=3:
            candidate.append(paruto_leading_edge[x][1][0])
            candidate.append(paruto_leading_edge[x][2][0])
            candidate.append(paruto_leading_edge[x][0][0])
        elif len(paruto_leading_edge[x])==2:
            candidate.append(paruto_leading_edge[x][0][0])
            candidate.append(paruto_leading_edge[x][1][0])
        elif len(paruto_leading_edge[x])==1:
            candidate.append(paruto_leading_edge[x][0][0])
        candidate = list(set(candidate))
        random.shuffle(candidate)
        candidates.append(candidate)
    l = []
    for i in range(n):
        l.append(i)
        random.shuffle(l)
    """
    The initial path is optimized by perturbation
    """
    for i in l:
        for j in range(len(candidates[i])):
            t = copy.deepcopy(best_add)
            t[i] = candidates[i][j]
            S = copy.deepcopy(M)
            for p in t:
                S.append(points[p])
            sol, sumdis = LKH(np.array(S))
            tour = []
            for e in range(1, len(sol)):
                tour.append([sol[e - 1], sol[e]])
            if sumdis<best_fitness:
                best_fitness=copy.deepcopy(sumdis)
                best_add = copy.deepcopy(t)
                best_points = copy.deepcopy(S)
                best_tour = copy.deepcopy(tour)
            else:
                continue
    """
    Draw the final path of the algorithm
    """
    plt_points(np.array(best_points))
    plt_edges(best_tour, np.array(best_points), color='k')
    bfc = best_fitness
    plt.title(int(bfc))
    plt.show()
    points_r = turn_points(best_points, points)
    pt = turn_tour(best_tour, points_r)
    print(f"length of path = {bfc}")
    print(f"path{pt}")

if __name__ == '__main__':
    """
    You can replace fileName with any of the data sets in the following list.
    Specifically, you can select any dataset in the dataset/TSPLIB folder
    ["pr76.csv","kroA100.csv","kroC100.csv","pr124.csv","pr136.csv","ch150.csv","kroA150.csv"]
    This file records the coordinate information of each point.
    """
    fileName = "kroA150.csv"


    """
    You can replace percent with any of the values in the list below.
    [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    """
    percent = 0.4               #|M|=percent*(Number of nodes in data set fileName).M stands for mandatory node set.
    semantics = 5               #There are 5 semantics in the data set

    SRC_LKH_main(fileName,percent,semantics)





