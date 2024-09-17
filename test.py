from utils import one_azimuth_scan, makeWall, plot

import pickle
import matplotlib.pyplot as plt

import time

import numpy as np

filename = "./data"

with open(filename, 'rb') as f:
    data = pickle.load(f)

robot_list = data['robot_list']
human_list = data['human_list']

robot = robot_list[0]

mapinfo = makeWall(robot.position, 51)

dst = [40, 100]

ogm = np.ones((100, 100))
fig, ax = plt.subplots()
isShowAxisOption=False
for dst in mapinfo:
        plt.cla()
        ogm_info = {"size": [100, 100]}
        agent_value_list = [[agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info] for agent in human_list]
        a = one_azimuth_scan(ogm_info, robot.ogm.mapinfo, robot.position, dst, agent_value_list, 50)
        
        plot(a[0], fig, ax, mapinfo = robot.ogm.mapinfo, agent_list=human_list, dst=dst, now=robot.position, axis_option=not isShowAxisOption)
        ogm *= a[0]
        isShowAxisOption = True
        plt.pause(0.05)

plt.close()

fig, ax = plt.subplots()
plot(ogm, fig, ax, mapinfo = robot.ogm.mapinfo, agent_list=human_list, dst=dst, now=robot.position, axis_option=not isShowAxisOption)

plt.show()
