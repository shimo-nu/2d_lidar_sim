from utils import one_azimuth_scan, makeWall, plot

import pickle
import matplotlib.pyplot as plt

import time

import numpy as np

from agents.human import HumanManager


robot_position = [50, 50]

agent_positions_by_pattern = [[70, 50, 0], [40, 50, 0], [80, 45, np.pi/4], [35, 20, np.pi/2], [65, 30, np.pi/3], [80, 30, np.pi/2]]
human_list = HumanManager()
human_list.nomove(agent_positions_by_pattern)

mapinfo = makeWall(robot_position, 51)

new_mapinfo = [
    [0, 0],
    [100, 0],
    [100, 60],
    [40, 60],
    [40, 75],
    [100, 75],
    [100, 100],
    [0, 100],
    [0, 0]
]

dst = [40, 100]

ogm = np.ones((100, 100))
fig, ax = plt.subplots()
isShowAxisOption=False
for dst in mapinfo:
        plt.cla()
        ogm_info = {"size": [100, 100]}
        agent_value_list = [[agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info] for agent in human_list]
        a = one_azimuth_scan(ogm_info, new_mapinfo, robot_position, dst, agent_value_list, 50)
        
        plot(a[0], fig, ax, mapinfo = new_mapinfo, agent_list=human_list, dst=dst, now=robot_position, axis_option=not isShowAxisOption)
        ogm *= a[0]
        isShowAxisOption = True
        plt.pause(0.05)

plt.close()

fig, ax = plt.subplots()
plot(ogm, fig, ax, mapinfo = new_mapinfo, agent_list=human_list, dst=dst, now=robot_position, axis_option=not isShowAxisOption)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.show()
