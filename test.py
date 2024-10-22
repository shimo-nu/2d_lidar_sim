from utils import lidar_scan, one_azimuth_scan, makeWall, plot, TimeMeasure

import pickle
import matplotlib.pyplot as plt

import time

import numpy as np

from agents.human import HumanManager


robot_position = [50, 50]

agent_positions_by_pattern = [
    [70, 50, 0],         # 1人目
    [40, 50, 0],         # 2人目
    [80, 45, np.pi/4],   # 3人目
    [35, 20, np.pi/2],   # 4人目
    [65, 30, np.pi/3],   # 5人目
    [80, 30, np.pi/2],   # 6人目
    [75, 40, np.pi/6],   # 7人目
    [45, 45, -np.pi/6],  # 8人目
    [60, 35, np.pi/4],   # 9人目
    [50, 40, -np.pi/3],  # 10人目
    [55, 50, 0],         # 11人目
    [65, 55, -np.pi/4],  # 12人目
    [70, 40, np.pi/2],   # 13人目
    [60, 60, -np.pi/2],  # 14人目
]
neighbour_target = np.array([5, 50])

human_manager = HumanManager()
human_manager.rvo2setup(agent_positions_by_pattern, neighbour_target=neighbour_target)
print(f"Human Number is {len(human_manager)}")
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

ogm = np.ones((100, 100))
fig, ax = plt.subplots()
isShowAxisOption=False
isAnimation = False

tm = TimeMeasure(3)
for dst in mapinfo:
        ogm_info = {"size": [100, 100]}
        agent_value_list = [[agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info] for agent in human_manager]
        tm.from_cp(1)
        a = one_azimuth_scan(ogm_info, new_mapinfo, robot_position, dst, agent_value_list, 50)
        tm.to_cp(1)
        if (isAnimation):
            plot(a[0], fig, ax, mapinfo = new_mapinfo, agent_list=human_manager, dst=dst, now=robot_position, axis_option=not isShowAxisOption)
            plt.pause(0.05)
            plt.cla()
        ogm *= a[0]
        isShowAxisOption = True

plt.close()

tm.result()

tm = TimeMeasure(1)

tm.from_cp(1)
lidar_scan(robot_position, ogm_info["size"], new_mapinfo, 50, human_manager)
tm.to_cp(1)
tm.result()
fig, ax = plt.subplots()
plot(ogm, fig, ax, mapinfo = new_mapinfo, agent_list=human_manager, dst=dst, now=robot_position, axis_option=not isShowAxisOption)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.show()
