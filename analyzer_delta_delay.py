import numpy as np
import csv
import time
from skyfield.api import wgs84
import gurobipy as gp
from gurobipy import Model, GRB
import math
import random
import multiprocessing as mp
import pandas as pd

light_speed = 299.792458  # 单位km/ms

ue_pos = []  # 用户blh位置 lat lon alt
ue_pos_cbf = []  # 用户cbf位置
est_ue_pos = []  # 估计的blh位置
est_ue_pos_cbf = []  # 估计的cbf位置
difdis = []  # 误差——计算cbf位置与目标cbf位置距离


def to_cbf(lat, lon, alt):
    '''
    transform (latitude, longitude, altitude) to cbf
    '''
    cbf = list(wgs84.latlon(lat, lon, alt * 1000).itrs_xyz.km)  # 函数要求传入的海拔单位为m
    return cbf

def to_blh(x, y, z):
    a = 6378.137  # 地球的半长轴 单位km
    f = 1 / 298.257223563  # 扁率
    e2 = 2 * f - f ** 2  # 第一偏心率平方
    b = a * np.sqrt(1 - e2)  # 半短轴
    e2_prime = e2 / (1 - e2)  # 第二偏心率平方
    
    lon = np.arctan2(y, x)  # 单位弧度

    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * a, p * b)
    phi = np.arctan2(z + e2_prime * b * np.sin(theta) ** 3,
                    p - e2 * a * np.cos(theta) ** 3)

    # 计算曲率半径和高度
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    alt = (p / np.cos(phi) - N) / 1000  # 单位km

    # 转换单位度
    lat = np.degrees(phi)
    lon = np.degrees(lon)

    return [lon, lat, alt]

def cal_dis(sat_cbf, pt_cbf):
    return np.sqrt(np.square(sat_cbf[0]-pt_cbf[0])+np.square(sat_cbf[1]-pt_cbf[1])+np.square(sat_cbf[2]-pt_cbf[2]))      


def get_ue_loc(loc_file):
    '''
    return a list of ue_loc [[-0.1, 51.5], ...] lat lon
    '''
    ue_loc = []
    ue_loc_cbf = []
    with open(loc_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ue_loc.append([float(row[0]), float(row[1]), 0])
            ue_loc_cbf.append(to_cbf(float(row[0]), float(row[1]), 0))
    return ue_loc, ue_loc_cbf

def init_paras(trace_file, ue_idx):
    df = pd.read_csv(trace_file)
    delta_dis = df['delta']*light_speed
    sat_pos_cbf = []  # 卫星前后时刻xyz坐标对[[[x1,y1,z1],[x2,y2,z2]], ...]
    for i in range(len(df)):
        sat_pos_cbf.append([[df['x1'][i], df['y1'][i], df['z1'][i]], [df['x2'][i], df['y2'][i], df['z2'][i]]])
    return np.array(delta_dis), np.array(sat_pos_cbf)

def gen_init_pt(ue_pos, set_dis):
    '''随机生成一个到ground truth固定距离的搜索起始点(cbf)'''
    bearing = random.uniform(0, 2 * math.pi)  # 随机生成方位角
    distance = set_dis  # 观测者距离ground truth 设置为10km
    lat, lon = ue_pos[0], ue_pos[1]
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance / 6371) +
                             math.cos(lat_rad) * math.sin(distance / 6371) * math.cos(bearing_rad))
    new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance / 6371) * math.cos(lat_rad),
                                       math.cos(distance / 6371) - math.sin(lat_rad) * math.sin(new_lat_rad))
    
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)
    init_pt_cbf = to_cbf(new_lat, new_lon, 0)

    print(cal_dis(to_cbf(ue_pos[0], ue_pos[1], 0), init_pt_cbf))
    return init_pt_cbf

def optimize_with_gurobi(sat_pos_cbf, delta_dis, initial_guess, timelimit, ue_num):

    model = gp.Model("UE_position_optimization")

    # 设置模型参数
    model.setParam('TimeLimit', timelimit/ue_num)  # 设置求解时间限制 
    model.setParam('MIPGap', 1e-3)    # 小于最优性容差视为解 
    model.setParam('MIPFocus', 1)  # 1 更关注找到解 2 更关注找到最优解
    model.setParam('Threads', 0)      # 设置使用8个线程 设置为0会自动调用空余线程
    model.setParam('Heuristics', 0.5) # 设置启发式方法的频率/强度
    model.setParam('FeasibilityTol', 0.0001)  # 可行解的精度限制 最小1e-9 
    model.setParam('OptimalityTol', 0.00000001)  # 最优解的精度限制 最小1e-9

    xbounds = [-6379, 6379]
    ybounds = [-6379, 6379]
    zbounds = [-6379, 6379]

    # 决策变量
    x = model.addVar(lb=xbounds[0], ub=xbounds[1], name="x")
    y = model.addVar(lb=ybounds[0], ub=ybounds[1], name="y")
    z = model.addVar(lb=zbounds[0], ub=zbounds[1], name="z")
    x.Start = initial_guess[0]
    y.Start = initial_guess[1]
    z.Start = initial_guess[2]

    dist_var_1 = []
    dist_vars_1 = []  # 点到时刻1卫星位置距离平方
    dist_sq_var_1 = []
    dist_sq_vars_1 = []  # 点到时刻1卫星位置距离差
    dist_var_2 = []
    dist_vars_2 = []  # 点到时刻2卫星位置距离平方
    dist_sq_var_2 = []
    dist_sq_vars_2 = []  # 点到时刻2卫星位置距离差
    delta_dist_var = []
    delta_dist_vars = []  # 距离2与距离1差值与真实delta_dis之差
    #delta_abs_var = []
    #delta_abs_vars = []  # 距离2与距离1差值与真实delta_dis之差绝对值

    for i in range(len(sat_pos_cbf)):
        dist_var_1 = model.addVar(lb=0, ub=GRB.INFINITY)  
        dist_vars_1.append(dist_var_1)
        dist_sq_var_1 = model.addVar(lb=0, ub=GRB.INFINITY)  
        dist_sq_vars_1.append(dist_sq_var_1)

        dist_var_2 = model.addVar(lb=0, ub=GRB.INFINITY)  
        dist_vars_2.append(dist_var_2)
        dist_sq_var_2 = model.addVar(lb=0, ub=GRB.INFINITY)  
        dist_sq_vars_2.append(dist_sq_var_2)

        delta_dist_var = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)  
        delta_dist_vars.append(delta_dist_var)
        #delta_abs_var = model.addVar(lb=0, ub=GRB.INFINITY)
        #delta_abs_vars.append(delta_abs_var)


    for i, (sat_pos_pair, obs_delta_dis) in enumerate(zip(sat_pos_cbf, delta_dis)):
        model.addConstr(dist_vars_1[i] == (sat_pos_pair[0][0]-x)**2 + (sat_pos_pair[0][1]-y)**2 + (sat_pos_pair[0][2]-z)**2)
        model.addConstr(dist_vars_2[i] == (sat_pos_pair[1][0]-x)**2 + (sat_pos_pair[1][1]-y)**2 + (sat_pos_pair[1][2]-z)**2)
        model.addConstr(dist_sq_vars_1[i]**2 == dist_vars_1[i])
        model.addConstr(dist_sq_vars_2[i]**2 == dist_vars_2[i])

        model.addConstr(delta_dist_vars[i] == dist_sq_vars_2[i] - dist_sq_vars_1[i] - obs_delta_dis)
        #model.addConstr(delta_abs_vars[i] >= delta_dist_vars[i])
        #model.addConstr(delta_abs_vars[i] >= -delta_dist_vars[i])

    model.addConstr((initial_guess[0]-x)**2 + (initial_guess[1]-y)**2 + (initial_guess[2]-z)**2 <= 24**2)

    ### 构建目标函数
    #objective = gp.quicksum(delta_abs_vars[i]**2 for i in range(len(delta_abs_vars)))
    objective = gp.quicksum(delta_dist_vars[i]**2 for i in range(len(delta_dist_vars)))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        ue_x, ue_y, ue_z = x.X, y.X, z.X
        return [ue_x, ue_y, ue_z]

def get_pos(trace_file, est_ue_loc_file, ue_pos, ue_pos_cbf, ue_idx, timelimit, ue_num):

    delta_dis, sat_pos_cbf = init_paras(trace_file, ue_idx)
    initial_guesses = gen_init_pt(ue_pos, 10)

    with open(est_ue_loc_file, 'a', newline='') as file:
        writer = csv.writer(file)            
                    
        t = time.time()
        result = optimize_with_gurobi(sat_pos_cbf, delta_dis, initial_guesses, timelimit, ue_num)

        if result:
            est_ue_pos.append(to_blh(result[0], result[1], result[2]))
            est_ue_pos_cbf.append(to_cbf(est_ue_pos[-1][1], est_ue_pos[-1][0], 0))
            difdis.append(cal_dis(ue_pos_cbf, est_ue_pos_cbf[-1])) 
            writer.writerow(['lon', 'lat', 'x', 'y', 'z', 'difdis', 'time', 'ue_idx'])
            writer.writerow([est_ue_pos[-1][0], est_ue_pos[-1][1], est_ue_pos_cbf[-1][0], est_ue_pos_cbf[-1][1], est_ue_pos_cbf[-1][2], difdis[-1], time.time()-t, ue_idx])
            print("Estimated UE position in CBF:", est_ue_pos_cbf[-1])
            print("Real UE position in CBF:", ue_pos_cbf)
            print("Estimated UE position in BLH:", est_ue_pos[-1])
            print("Real UE position in BLH:", ue_pos)
            print("Difference in distance:", difdis[-1])
        else:
            writer.writerow(['lon', 'lat', 'x', 'y', 'z', 'difdis', 'time', 'ue_idx'])
            writer.writerow([0, 0, 0, 0, 0, 0, 0, 0])
            print("Optimization failed to find a solution.")


if __name__ == "__main__":
    
    timelimit = 60 # second
    ue_num = 1 # number of users
    ue_loc, ue_loc_cbf = get_ue_loc('examples/ue_loc.csv')
    trace_file = 'examples/trace_delta_delay.csv'
    est_ue_loc_file = 'examples/result_delta_delay.csv'
    
    pool = mp.Pool(processes=mp.cpu_count()) 
    
    tasks = []
    for ue_idx in range(ue_num):
        ue_pos = ue_loc[ue_idx]
        ue_pos_cbf = ue_loc_cbf[ue_idx]
        task = get_pos(trace_file, est_ue_loc_file, ue_pos, ue_pos_cbf, ue_idx, timelimit, ue_num)
        tasks.append(task)

    # 使用进程池并行执行任务
    res = pool.map_async(get_pos, tasks)


