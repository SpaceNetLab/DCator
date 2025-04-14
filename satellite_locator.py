import skyfield
from skyfield.api import load, wgs84, EarthSatellite, utc
from skyfield.framelib import itrs
from datetime import datetime, date
import time
import numpy as np
import pandas as pd

def cal_distance(sat_cbf, ue_cbf):
    # calculate the distance between a satellite and a UE
    # sat_cbf: 1d-array, ue_cbf: 1d-array
    dis = np.sqrt(np.sum(np.square(sat_cbf-ue_cbf)))        
    return dis

def read_tle(filename):
    # read TLE file, return a list of satellites
    f = open(filename)
    sats = []
    while True:
        sat_id = f.readline().strip()
        if sat_id=='': break
        line1 = f.readline().strip()
        line2 = f.readline().strip()
        sats.append(EarthSatellite(line1, line2, sat_id))
        
    f.close()

    return sats

def extract_frequency(filename):
    # extract frequency from the *.parsed file
    f = open(filename)
    start_time = -1
    freqs = []
    while True:
        line = f.readline().strip()
        if line=='': break
        data = line.split()
        if not data[1].startswith('p-'): continue
        if start_time==-1:
            start_time = int(data[1].split('-')[1])
        time_ms = float(data[2])
        freq = int(data[3])
        confidence = int(data[4][:-1])
        lcw = "LCW" in line
        freqs.append([time_ms,freq,confidence,lcw])
        
    f.close()
    return start_time,freqs

t0 = time.time()
ts = load.timescale()

# change the TLE file
sats = read_tle('TLE/Iridium.txt')

# change the raw trace file
df = pd.read_csv('examples/trace_raw.csv')

# change the *.parsed file
start_time,freqs = extract_frequency('examples/output.parsed')

# change the location of satellite locator
ue_loc = [40.0038889,116.3258333]


ue_cbf = wgs84.latlon(ue_loc[0], ue_loc[1], 0).itrs_xyz.km
# print(ue_cbf)

datetime_format = '%Y-%m-%dT%H:%M:%S.%f'
rx_time = []
dts = []
l = len(df)
for i in range(l):
    rx_time.append(datetime.strptime(df['rx_time'][i],datetime_format))
    dt = datetime.strptime(df['time'][i],datetime_format)
    dt = dt.replace(tzinfo=utc)
    dts.append(dt)
#print(dts)
t_ts = ts.from_datetimes(dts)


# time zone sync
start_time -= 28800


light_speed = 299.792458 # km/ms

real_delay = []
real_doppler = []

result = []
for t in range(l):
    res = []
    sel_lat = 0
    sel_cbf = []
    sel_vel = []
    for i in range(len(sats)):
        sat = sats[i]
        geocentric = sat.at(t_ts[t])
        subpoint = wgs84.subpoint(geocentric)
        sat_lat = subpoint.latitude.degrees
        sat_lon = subpoint.longitude.degrees
        sat_alt = subpoint.elevation.km
        #print(sat_lat,sat_lon,sat_alt)


        #sat_cbf = to_cbf(sat_lat, sat_lon, sat_alt)
        #print(sat_cbf)
        #print(subpoint.itrs_xyz.km)
        itrs_xyz_vel = geocentric.frame_xyz_and_velocity(itrs)
        sat_cbf = itrs_xyz_vel[0].km
        sat_vel = itrs_xyz_vel[1].km_per_s
        #print(np.sqrt(np.sum(sat_vel*sat_vel)))
        dis = cal_distance(sat_cbf, ue_cbf)
        latency = dis/light_speed
        #print(latency)
        
        
        
        if abs(latency-df['latency'][t])<0.02:
            #print('latency',t, i, latency, df['latency'][t]-latency, (df['latency'][t]-latency)/latency)
            res = [sat_cbf[0],sat_cbf[1],sat_cbf[2],sat_lon,sat_lat,sat_alt,df['latency'][t],sat_vel[0],sat_vel[1],sat_vel[2],df['doppler'][t]]
            sel_lat = latency
            sel_cbf = sat_cbf
            sel_vel = sat_vel
            real_delay.append(latency)
            #print(sat_cbf)

    
    delta_time =  (rx_time[t].timestamp()-start_time)*1000
    
    # select the frequency with the highest confidence
    index = -1
    for j in range(len(freqs)):
        if abs(delta_time-freqs[j][0])<10 and freqs[j][3] and (index==-1 or freqs[index][2]<=freqs[j][2]):
            index = j
    freq = freqs[index][1]
    
    doppler_shift = np.sum(sel_vel*(ue_cbf-sel_cbf))/(sel_lat*light_speed)*freq/light_speed/1e3
    #print('doppler',freq,df['doppler'][t]-doppler_shift,(df['doppler'][t]-doppler_shift)/doppler_shift*100,freqs[index][2])
    real_doppler.append(doppler_shift)        
    res.append(freq)
    result.append(res)

#real_delay = np.array(real_delay)
#real_doppler = np.array(real_doppler)
#print((df['latency']-real_delay)/real_delay*100)
#print((df['doppler']-real_doppler)/real_doppler*100)

result = pd.DataFrame(result,columns=['x','y','z','lon','lat','alt','delay','vx','vy','vz','doppler','freq'])

# gen delay trace
result_delay = result[['x','y','z','delay']]
result_delay.to_csv('examples/trace_delay_1.csv', index=False)

# gen delta delay trace
result_delta_delay = []
for t in range(l-1):
    result_delta_delay.append([result['x'][t],result['y'][t],result['z'][t],result['x'][t+1],result['y'][t+1],result['z'][t+1],result['delay'][t+1]-result['delay'][t]])
result_delta_delay = pd.DataFrame(result_delta_delay,columns=['x1','y1','z1','x2','y2','z2','delta'])
result_delta_delay.to_csv('examples/trace_delta_delay_1.csv', index=False)

# gen Doppler trace
result_doppler = result[['x','y','z','vx','vy','vz','freq','doppler']]
result_doppler.to_csv('examples/trace_doppler_1.csv', index=False)