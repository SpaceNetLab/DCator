import pandas as pd

# extract delay, Doppler shift from collected message file
# change the collected file
f = open('examples/message.txt')
rx_time = []
time = []
latency = []
doppler = []
last_time =''
while True:
    line = f.readline().strip()
    if line=='': break
    data = line.split()
    # read Access Request Message
    if len(data)>5 and data[3]=='[0600]':
        rx_t = data[0][:-1]
        ds = data[15][3:9]
        lat = data[16][3:8]
        t = data[17][2:-1]
        if t!=last_time:
            print(t,lat,ds)
            last_time = t
            rx_time.append(rx_t)
            time.append(t)
            latency.append(lat[1]+'.'+lat[2:])
            doppler.append(ds)
f.close()
result = pd.DataFrame()
result['rx_time'] = rx_time
result['time'] = time
result['latency'] = latency
result['doppler'] = doppler
result.to_csv('examples/trace_raw.csv',index=False)