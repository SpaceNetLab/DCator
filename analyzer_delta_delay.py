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

light_speed = 299.792458  # km/ms

ue_pos = []  
ue_pos_cbf = []  
est_ue_pos = []  
est_ue_pos_cbf = []  
difdis = []  


def to_cbf(lat, lon, alt):
    '''
    transform (latitude, longitude, altitude) to cbf
    '''
    cbf = list(wgs84.latlon(lat, lon, alt * 1000).itrs_xyz.km)  
    return cbf

def to_blh(x, y, z):
    a = 6378.137  
    f = 1 / 298.257223563  
    e2 = 2 * f - f ** 2  
    b = a * np.sqrt(1 - e2)  
    e2_prime = e2 / (1 - e2)  
    
    lon = np.arctan2(y, x)  

    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * a, p * b)
    phi = np.arctan2(z + e2_prime * b * np.sin(theta) ** 3,
                    p - e2 * a * np.cos(theta) ** 3)

    
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    alt = (p / np.cos(phi) - N) / 1000  

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

def init_paras(trace):
    delta_dis = trace['delta']*light_speed
    sat_pos_cbf = trace[['x1','y1','z1','x2','y2','z2']]
    return np.array(delta_dis), np.array(sat_pos_cbf)

def gen_init_pt(ue_pos, set_dis):
    # randomly generate the location of satellite locator for large-scale simulation
    bearing = random.uniform(0, 2 * math.pi)  
    distance = set_dis  
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

    # model parameters
    model.setParam('TimeLimit', timelimit/ue_num) 
    model.setParam('MIPGap', 1e-3)    
    model.setParam('MIPFocus', 1)  
    model.setParam('Threads', 0)      
    model.setParam('Heuristics', 0.5) 
    model.setParam('FeasibilityTol', 1e-3)  
    model.setParam('OptimalityTol', 1e-3)  

    xbounds = [-6379, 6379]
    ybounds = [-6379, 6379]
    zbounds = [-6379, 6379]

    # variables
    x = model.addVar(lb=xbounds[0], ub=xbounds[1], name="x")
    y = model.addVar(lb=ybounds[0], ub=ybounds[1], name="y")
    z = model.addVar(lb=zbounds[0], ub=zbounds[1], name="z")
    x.Start = initial_guess[0]
    y.Start = initial_guess[1]
    z.Start = initial_guess[2]

    dist_var_1 = []
    dist_vars_1 = []  
    dist_sq_var_1 = []
    dist_sq_vars_1 = []  
    dist_var_2 = []
    dist_vars_2 = []  
    dist_sq_var_2 = []
    dist_sq_vars_2 = []  
    delta_dist_var = []
    delta_dist_vars = []  

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
        

    for i, (sat_pos_pair, obs_delta_dis) in enumerate(zip(sat_pos_cbf, delta_dis)):
        model.addConstr(dist_vars_1[i] == (sat_pos_pair[0]-x)**2 + (sat_pos_pair[1]-y)**2 + (sat_pos_pair[2]-z)**2)
        model.addConstr(dist_vars_2[i] == (sat_pos_pair[3]-x)**2 + (sat_pos_pair[4]-y)**2 + (sat_pos_pair[5]-z)**2)
        model.addConstr(dist_sq_vars_1[i]**2 == dist_vars_1[i])
        model.addConstr(dist_sq_vars_2[i]**2 == dist_vars_2[i])

        model.addConstr(delta_dist_vars[i] == dist_sq_vars_2[i] - dist_sq_vars_1[i] - obs_delta_dis)
        
    model.addConstr((initial_guess[0]-x)**2 + (initial_guess[1]-y)**2 + (initial_guess[2]-z)**2 <= 24**2)

    # optimization target
    objective = gp.quicksum(delta_dist_vars[i]**2 for i in range(len(delta_dist_vars)))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        ue_x, ue_y, ue_z = x.X, y.X, z.X
        return [ue_x, ue_y, ue_z]

def get_pos(trace, est_ue_loc_file, ue_pos, ue_pos_cbf, ue_idx, timelimit, ue_num):

    delta_dis, sat_pos_cbf = init_paras(trace)
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
    
    timelimit = 180 # second
    ue_num = 200 # number of users
    ue_loc, ue_loc_cbf = get_ue_loc(f'examples/ue_loc_{ue_num}.csv')
    trace_file = f'examples/trace_delta_delay_{ue_num}.csv'
    est_ue_loc_file = f'examples/result_delta_delay_{ue_num}.csv'
    
    trace = pd.read_csv(trace_file)
    l = len(trace)//ue_num

    pool = mp.Pool(processes=mp.cpu_count()) 
    
    tasks = []
    for ue_idx in range(ue_num):
        ue_pos = ue_loc[ue_idx]
        ue_pos_cbf = ue_loc_cbf[ue_idx]
        task = get_pos(trace.loc[l*ue_idx:l*(ue_idx+1)-1], est_ue_loc_file, ue_pos, ue_pos_cbf, ue_idx, timelimit, ue_num)
        tasks.append(task)

    res = pool.map_async(get_pos, tasks)


