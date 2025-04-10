import math
import skyfield
from skyfield.api import load, wgs84, EarthSatellite, Topos
from skyfield.framelib import itrs
from datetime import datetime, date, time, timezone,timedelta
import time
import numpy as np
import pandas as pd
import random
import csv
import os


light_speed = 299.792458 # km/ms

def read_tle(filename, tle_type):
    '''
    1: all satellites
    2: DTC satellites
    3: non-DTC satellites
    '''
    sats = []
    with open(filename) as f:
        while True:
            sat_id = f.readline().strip()
            if sat_id=='': break
            line1 = f.readline().strip()
            line2 = f.readline().strip()
            if tle_type==1 or (tle_type==2 and 'DTC' in sat_id) or (tle_type==3 and not 'DTC' in sat_id):
                sats.append(EarthSatellite(line1, line2, sat_id))

    return sats

def cal_dis(sat_cbf, ue_cbf):
    return np.sqrt(np.sum(np.square(sat_cbf-ue_cbf),axis=1))      

def to_cbf(lat, lon, alt):
    '''
    transform (latitude, longitude, altitude) to cbf
    '''
    cbf = list(wgs84.latlon(lat, lon, alt * 1000).itrs_xyz.km)  # unit of m
    return cbf

def gen_loc(ue_loc_file, ue_num):
    # randomly generate ue_num locations, save in ue_loc_file
    ue_loc = []
    for _ in range(ue_num):
        # set the range of UE location, you can change it to fit the coverage of a constellation
        lat = random.uniform(25, 50)
        lon = random.uniform(-130, -70)
        ue_loc.append([lat, lon])
    ue_loc = pd.DataFrame(ue_loc, columns=['lat','lon'])
    ue_loc.to_csv(ue_loc_file, index=False)
    return

def gen_trace(ue_loc_file, ue_num, interval, duration, tle_file, tle_type):
    # generate traces of delay, delta delay and Doppler shift clues
    t0 = time.time()
    ts = load.timescale()
    sats = read_tle(tle_file,tle_type)  # read TLE data
    ue_locs = pd.read_csv(ue_loc_file)  # read UE locations

    # set the time of simulation
    t_ts = ts.utc(2025, 4, 8, 0, 0, range(0, duration, interval))
    sats_info = []
    for idx, sat in enumerate(sats):
        geocentric = sat.at(t_ts)
        subpoint = wgs84.subpoint(geocentric)
        sat_info = {
            'lat': subpoint.latitude.degrees, 
            'lon': subpoint.longitude.degrees,
            'alt': subpoint.elevation.km,
            'cbf': subpoint.itrs_xyz.km.T,
            'vel': geocentric.frame_xyz_and_velocity(itrs)[1].km_per_s.T, 
        }
        sats_info.append(sat_info)



    trace_delay = []
    trace_delta_delay = []
    trace_doppler = []
    for ue_idx in range(ue_num):
        ue_cbf = wgs84.latlon(ue_locs['lat'][ue_idx], ue_locs['lon'][ue_idx], 0).itrs_xyz.km

        latencies = []
        for idx in range(len(sats_info)):
            dis = cal_dis(sats_info[idx]['cbf'], ue_cbf)
            latency = dis/light_speed # ms
            latencies.append(latency)   
        latencies = np.array(latencies)
        # select the nearest satellite to offer service
        connected_sat = latencies.argmin(axis=0)    
        
        for time_idx in range(duration//interval):
            s = connected_sat[time_idx]
            x = sats_info[s]['cbf'][time_idx][0]
            y = sats_info[s]['cbf'][time_idx][1]
            z = sats_info[s]['cbf'][time_idx][2]
            delay = latencies[s][time_idx]
            vx = sats_info[s]['vel'][time_idx][0]
            vy = sats_info[s]['vel'][time_idx][1]
            vz = sats_info[s]['vel'][time_idx][2]
            freq = 1622000000 # set frequency
            dop = sum(sats_info[s]['vel'][time_idx]*(ue_cbf-sats_info[s]['cbf'][time_idx])*freq/(delay*light_speed*light_speed*1e3)) 

            # simulate random systematic error 
            delay = delay * (1+ random.uniform(-0.05, 0.05)/100)
            dop = dop * (1+ random.uniform(-0.05, 0.05)/100)
        
            if time_idx>0:
                trace_delta_delay.append([trace_delay[-1][0],trace_delay[-1][1],trace_delay[-1][2],x,y,z,delay-trace_delay[-1][3]])
            trace_delay.append([x,y,z,delay])
            trace_doppler.append([x,y,z,vx,vy,vz,freq,dop])
    trace_delay = pd.DataFrame(trace_delay, columns=['x','y','z','delay'])
    trace_delay.to_csv(f'examples/trace_delay_{ue_num}.csv', index=False)
    trace_delta_delay = pd.DataFrame(trace_delta_delay, columns=['x1','y1','z1','x2','y2','z2','delta'])
    trace_delta_delay.to_csv(f'examples/trace_delta_delay_{ue_num}.csv', index=False)
    trace_doppler = pd.DataFrame(trace_doppler, columns=['x','y','z','vx','vy','vz','freq','doppler'])
    trace_doppler.to_csv(f'examples/trace_doppler_{ue_num}.csv', index=False)


if __name__ == "__main__":
    # set the number of users
    ue_num = 200
    # set the user location file
    ue_loc_file = f'examples/ue_loc_{ue_num}.csv'
    if not os.path.exists(ue_loc_file): gen_loc(ue_loc_file=ue_loc_file, ue_num=ue_num)

    tle_type = 1 # use all satellites
    tle_file = 'TLE/Iridium.txt'
    interval = 60 # interval of ghost calls
    duration = 60*15 # monitoring duration
    gen_trace(ue_loc_file, ue_num, interval, duration, tle_file, tle_type)
    