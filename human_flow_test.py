from utils import one_azimuth_scan, makeWall, plot, TimeMeasure, lidar_scan

import pickle
import matplotlib.pyplot as plt

import time

import numpy as np

from agents.human import HumanManager
from rich.progress import track


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
    # [100, 60],
    # [40, 60],
    # [40, 75],
    # [100, 75],
    # [100, 100],
    [100, 100],
    [0, 100]
]

ogm = np.ones((100, 100))
fig, ax = plt.subplots()
isShowAxisOption=False
isAnimation = True
steps = 500

ogm_list = []
agent_positions = {agent.unique_num: {0: agent.position} for agent in human_manager}
ogm_info = {"size": [100, 100]}
tm = TimeMeasure(3)
for i in track(range(1, steps), description="Simulating..."):
    tm.setCPName(1, "lidar_scan")
    tm.from_cp(1)
    ogm = lidar_scan(robot_position, [100, 100], new_mapinfo, 50, human_manager)
    tm.to_cp(1)
    ogm_list.append(ogm)
    # plot(ogm, fig, ax, mapinfo = new_mapinfo, title=f"Step : {i}", agent_list=human_manager, now=robot_position, axis_option=not isShowAxisOption)
    # plt.pause(0.01)
    # plt.cla()
    tm.setCPName(2, "human move")
    tm.from_cp(2)
    human_manager.move(step=i)
    for human in human_manager:
        agent_positions[human.unique_num][i] = human.position
    tm.to_cp(2)
        
tm.result()
    
for i in range(1, steps - 1):
    for unique_num, position in agent_positions.items():

        human_manager.get(f"{unique_num}").position = position[i]
    plot(ogm_list[i], fig, ax, mapinfo = new_mapinfo, title=f"Step : {i}", agent_list=human_manager, now=robot_position, axis_option=not isShowAxisOption)
    plt.pause(0.1)
    plt.cla()
    isShowAxisOption = True
    
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.close()
