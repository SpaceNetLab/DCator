# DCator

## Prerequisites

DCator is run in python3 and the following libraries should be installed:
- numpy
- pandas
- skyfield
- gurobipy

DCator uses [gr-iridium](https://github.com/muccc/gr-iridium), [iridium-tools](https://github.com/muccc/iridium-toolkit), [LTESniffer](https://github.com/SysSec-KAIST/LTESniffer) as sniffers to obtain the location clues. 

## Usage

### Extracting location clues

Use the sniffer to collect data and extract the location clues in ```.csv``` files with the following fields:

- delay-based method: position of satellite and delay between satellite and UE
```
x,y,z,delay
```
- delta-delay-based method: positions of satellites and delay variation
```
x1,y1,z1,x2,y2,z2,delta
```
- Doppler-shift-based method: position and velocity of satellite, frequency and Doppler shift
```
x,y,z,vx,vy,vz,freq,doppler
```

DCator provides an example to extract location clues:
1. Use ```gr-iridium``` to collect ```.bits``` file.
2. Use ```iridium-parser.py``` in ```iridium-toolkit``` to generate ```.parsed``` file and use ```reassembler.py``` to decode the messages under ```idapp``` mode. We provide an example in ```examples/output.parsed``` and ```examples/message.txt```.
3. Download the TLE data from [NORAD](https://celestrak.org/NORAD/elements/). We provide an example in ```TLE/iridium.txt```.
4. Run ```extract_data.py``` to extract location clues and it generates ```examples/trace_raw.csv``` file:
```
python3 extract_data.py
```
1. Set the location of the satellite locator in ```satellite_locator.py``` and run it:
```
python3 satellite_locator.py
```
Finally, we get ```trace_delay_1.csv```, ```trace_delta_delay_1.csv```, ```trace_doppler_1.csv``` in ```examples```.

### Large-scale simulation

We provide a simulator to mimic large number of UEs in ```simulator.py```. It generates the locations of UEs in ```ue_loc_{ue_num}.csv``` and the traces of three types of location clues in ```examples```.

### Location inference

To infer the locations of UEs, DCator provides three analyzers: ```analyzer_delay.py```, ```analyzer_delta_delay.py```, ```analyzer_doppler.py```. Set the number of UEs and then run the analyzers to infer the locations based on different location clues:
```
python3 analyzer_delay.py
python3 analyzer_delta_delay.py
python3 analyzer_doppler.py
```
Results are shown in ```result_delay_{ue_num}.csv```, ```result_delta_delay_{ue_num}.csv```, ```result_doppler_{ue_num}.csv``` in ```examples``` respectively.