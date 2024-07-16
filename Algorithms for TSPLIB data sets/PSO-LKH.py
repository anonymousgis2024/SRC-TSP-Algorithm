
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

def readDataFile_list(fileName):
    """
    :param fileName: 输入一个文件的名字
    :return: 将数据集中的内容以(x,y)的形式返回
    """
    dataList = list()
    with open(r"../dataset/TSPLIB/"+fileName,'r') as fr:
        fr.readline()#跳过标题行
        for line in fr:
            strs = line.strip().split(",")
            dataList.append((float(strs[1]),float(strs[2])))
    return dataList

def types_m_init(n, left_point_num):
    """
    :param n:       需要将可选城市分为n类
    :param left_point_num:      可选城市的下标
    :return:        将可选城市分类存储
    """
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

class lz:
    def __init__(self,M,O):
        self.M = M
        self.O = O
        self.position = self.M+self.O
        self.fitness = float('inf')
        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_fitness = copy.deepcopy(self.fitness)
        self.pbest_O = O





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

#计算距离矩阵
def get_dis_mat(points,p_n):
    #将距离记录在矩阵C中
    dis_mat = []
    for i in range(p_n):
        dis_list = []
        for j in range(p_n):
            d = distance(points,i,j)
            dis_list.append(round(d,2))
        dis_mat.append(dis_list)
    # for i in dis_mat:
    #     print(i)
    return dis_mat

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
    return add_points

def count_fitness(l,dis_mat):
    point_list = l.position
    fitness = 0
    for i in range(1,len(point_list)):
        fitness+=dis_mat[point_list[i-1]][point_list[i]]
    fitness+=dis_mat[point_list[-1]][point_list[0]]
    return fitness

def find_gbest(lzq):
    gbest = lzq[0]
    # print(gbest.position)
    for l in lzq:
        if l.fitness<gbest.fitness:
            gbest = l
        else:
            continue
    return gbest

def exchange(parent1,parent2):
    start = random.randint(0, len(parent1) - 1)  # 交叉的起始位置
    end = random.randint(0, len(parent1) - 1)  # 结束位置
    if start > end:
        start, end = end, start
    choosed = parent2[start:end]
    left = list(set(parent1) - set(choosed))
    new_position = left[:start] + choosed + left[start:]
    return new_position

def get_y1_list(t1,t2,road,dis_mat,tour):
    y1_list = []
    dis = dis_mat[t1][t2]
    for p in tour:
        if (dis_mat[t2][p]<dis) and ([t2,p] not in road) and ([p,t2] not in road):
            y1_list.append([t2,p])
    return y1_list

def get_x_list(road,t3,t1):
    x2_list = []
    for e in road:
        if (t3 in e) and (t1 not in e):
            if e[1]==t3:
                e = e[::-1]
            x2_list.append(e)
    return x2_list

def count_dis(X,C):
    sum_dis = 0
    for e in X:
        f = e[0]
        t = e[1]
        sum_dis+=C[f][t]
    return sum_dis

def edit_Best_tour(X,Y,Best_tour):
    for e in X:
        if e in Best_tour:
            Best_tour.remove(e)
        elif e[::-1] in Best_tour:
            Best_tour.remove(e[::-1])
    for e in Y:
        Best_tour.append(e)
    return Best_tour

def not_same(X,Y):
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        if (x in Y) or (x[::-1] in Y) or (y in X) or (y[::-1] in X):
            return False
    else:
        return True

def do_2_opt(tour,dis_mat):
    road = []
    for i in range(1,len(tour)):
        road.append([tour[i-1],tour[i]])
    road.append([tour[-1],tour[0]])
    r = random.randint(0,len(road)-1)
    flag = 0
    x1 = road[r]
    t1 = x1[0]
    t2 = x1[1]
    y1_list = get_y1_list(t1,t2,road,dis_mat,tour)
    if len(y1_list)>0:
        for y1 in y1_list:
            t3 = y1[1]
            x2_list = get_x_list(road,t3,t1)
            for x2 in x2_list:
                X = [x1,x2]
                t4 = x2[1]
                y2 = [t4,t1]
                Y = [y1,y2]
                del_dis = count_dis(X, dis_mat)
                add_dis = count_dis(Y, dis_mat)
                if del_dis>add_dis:
                    if not_same(X,Y):
                        # print(X)
                        # print(Y)
                        # print(f"修改前road={road}")
                        road = edit_Best_tour(X,Y,road)
                        # print(f"修改后road={road}")
                        flag=1
                        break
                    else:
                        continue
                else:
                    continue
            if flag==1:
                break
    return road


def find_special_edge(start,road):
    for e in road:
        if start in e:
            return e
    else:
        return []

def new_Inertial_exploration(l, dis_mat):
    # print(f"经验探索开始,路径中的点数={len(l.position)}")
    # print("新方案")
    tour = l.position

    road = do_2_opt(tour,dis_mat)
    # print(len(road))
    point_num = list(set(element for sublist in road for element in sublist))
    # print(point_num)
    # if point_num!=len(road):
    #     print("####异常")
    new_position = []
    f = point_num[0]
    new_position.append(f)
    while len(road)>0:
        # print(new_position)
        # print(len(new_position))
        f = new_position[-1]
        sp_e = find_special_edge(f,road)
        if len(sp_e)>0 and len(set(sp_e)-{f})>0:
            t = (set(sp_e)-{f}).pop()
            new_position.append(t)
            road.remove(sp_e)
        else:
            break
    if len(new_position)<len(l.position):
        return []
    else:
        return new_position

def get_same_type_city(t,types):
    for i in range(len(types)):
        if t in types[i]:
            r = random.randint(0,len(types[i])-1)
            return types[i][r]
    else:
        return -1

def Inertial_exploration(l,types):
    """
    惯性探索:随机改变一个可选城市，用同类的城市替换它
    :param l:       粒子l，定义为一个class类
    :param types:   城市分类
    :return:        返回新的粒子
    """

    r = random.randint(0,len(l.O)-1)
    t = l.O[r]
    same_type_city = get_same_type_city(t,types)    #获得一个和城市t种类相同的城市
    if same_type_city!=-1:
        l.O[r] = same_type_city                     #将城市t替换掉
    l.position = l.M+l.O
    return l

def same_list(list1,list2):
    test1 = list(set(list1)-set(list2))
    test2 = list(set(list2)-set(list1))
    if len(test1)==0 and len((test2))==0:
        return 1
    else:
        return 0
def get_choose_city(list1,list2):
    list = []
    for i in range(len(list1)):
        a = random.randint(0, 1)
        if a==0:
            list.append(list1[i])
        elif a==1:
            list.append(list2[i])
    return list

def sort_by_types(list1,types):
    t = [None]*len(list1)
    for i in list1:
        for j in range(len(types)):
            if i in types[j]:
                t[j] = i
                continue
    return t

def Personal_experience_exploration(l):
    new_choose_city = get_choose_city(l.O,l.pbest_O)    #输入两个列表l.O和l.pbest_O,生成一个新列表，新列表中的第一个元素就是前面两个列表的第一个元素随机抽一个出来
    l.O = new_choose_city
    l.position = l.M+l.O
    return l



def Social_exploration(l,gbest):
    new_choose_city = get_choose_city(l.O,gbest.O)
    l.O = new_choose_city
    l.position = l.M+l.O
    return l

def find_a_city(f,dis_mat,left):
    dis = []
    for p in left:
        dis.append(dis_mat[f][p])
    min_dis = min(dis)
    index = dis.index(min_dis)
    return left[index]

def get_road(point_list,dis_mat):
    r = random.randint(0,len(point_list)-1)
    start = point_list[r]
    left = copy.deepcopy(point_list)
    left.remove(start)
    road = [start]
    while len(left)>0:
        f = road[-1]
        t = find_a_city(f,dis_mat,left)
        road.append(t)
        left.remove(t)
    return road

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

    # sol = elkai.solve_int_matrix(distance)
    sol = elkai.solve_float_matrix(distance,runs=10)#允许浮点距离
    sol.append(0)
    '''这个函数计算出的是有回路的TSP问题，但返回的解方案sol没有给出完整解方案(少一个终点0),故我们在代码中在解方案最末尾加上了编号0'''

    # print("最优解方案:", sol)
    sumdis = cal_fitness(sol)
    # print("最优解总长度:", sumdis)
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


def PSO_LKH_main(fileName,n,iterations,must_per,total_fitness,timeMax):
    """
    :param fileName:        Name of the data set
    :param n:               Semantic category
    :param iterations:      The maximum number of iterations
    :param must_per:        |M|=must_per*(Number of nodes in data set)
    :param total_fitness:   The target length of the path
    :param timeMax:         Maximum running time of the algorithm
    """

    m = 100     # The size of the particle swarm
    w = 1       # inertia factor
    c1 = 1      # Self-perception factor
    c2 = 1      # Social cognitive factor

    points = readDataFile_list(fileName)    #Read data set
    p_n = len(points)
    point_num = [i for i in range(p_n)]
    must_p_n = int(must_per*p_n)
    np.random.seed(0)
    must_points = list(np.random.choice(point_num, must_p_n, replace=False))    #Structural set M
    left_points= list(set(point_num)-set(must_points))                          #Remaining node
    types = types_m_init(n, left_points)        #The set S is divided into n classes according to semantics

    #Generate initial population
    lzq = []
    add_points = get_add_points(m,types)    #Multiparticle formation
    for a in add_points:
        l = lz(must_points,a)               #Generate a particle
        test_points_num = l.position
        test_points = []
        for i in test_points_num:
            test_points.append(points[i])
        sol, sumdis = LKH(np.array(test_points))  #The LKH algorithm is used to calculate the path corresponding to a certain particle
        l.fitness = l.pbest_fitness = sumdis
        lzq.append(l)

    gbest = copy.deepcopy(find_gbest(lzq))      #Find a global optimal solution from the initialized particle swarm, denoted gbest

    start_time = time.time()
    # Start iterative updates
    for i in range(iterations):
        """
        Each particle is iteratively updated according to the following formula
        V = wv+c1*random(0,1)*(pBest-X)+c2*random(0,1)*(gbest-X)
        """
        now = time.time()
        """If a better path than the target path is found, or the algorithm runs longer than online, it stops"""
        if gbest.fitness<=total_fitness+1 or now-start_time>timeMax:
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
            test_points = []
            for i in test_points_num:
                test_points.append(points[i])           #After each iteration, the fitness of each particle is recalculated
            sol, sumdis = LKH(np.array(test_points))
            l.fitness = sumdis
            if sumdis < l.pbest_fitness:                #Compared to itself, if a particle finds its shortest path so far, it updates the individual's optimal solution
                l.pbest_position = l.position
                l.pbest_fitness = sumdis
                l.pbest_O = l.O

    end_time = time.time()
    time_cost = end_time-start_time     #Calculate the running time of the algorithm.
    print(f"Time spent:{time_cost}s")
    # Plot the path determined by the algorithm
    test_points_num = gbest.position
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
    """This file records the coordinate information of each point"""
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

    PSO_LKH_main(fileName, semantics, iterations, percent, total, timeMax)


