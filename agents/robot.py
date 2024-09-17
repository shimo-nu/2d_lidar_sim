import os, sys
sys.path.append(os.pardir)

import numpy as np
import math
import matplotlib.pyplot as plt
import joblib
import copy
import io, logging
import random

import planner.global_planner.a_star as a_star
from planner.global_planner.a_star_bs import AStarBSPlanner 
from planner.global_planner.waypoint_generator import waypoint_generator, create_demand_layer_map, solver, sample_size_decider


from agents.agent import Agent
import planner.local_planner.dynamic_window_approach as dwa
from lib.utils import (one_azimuth_scan, createMask, calcEuclidean, Ellipse, makeWall, TimeMeasure, 
                       generate_grid_points_in_ellipse, circleRectangleCollision, 
                       point_in_polygon, is_point_inside_rectangle, is_agent_between_points, 
                       get_availble_cpus, is_projection_closer_to_a, searchDsts)
from gridmap import OccupancyGridMap
from lib.debug import debug_print
from prettytable import PrettyTable
from agents.agent_utils import LocalGoal
from planner.global_planner.grid_based_sweep_coverage_path_planner import planning as gbsweep_planning


class RobotManager():
    def __init__(self):
        self.robot_list = []
        self.allow_robot_name_duplicate = False
        self.robot_name_index = {}
        
        
    def __len__(self):
        return len(self.robot_list)
        
    def __getitem__(self, key):
        if (key > len(self.robot_list) - 1):
            raise IndexError("Index out of range")
        return self.robot_list[key]

    def append(self, robot):
        if robot.name not in self.robot_name_index:
            # self.robot_list[robot.name] = robot
            self.robot_list.append(robot)
            self.robot_name_index[robot.name] = len(self.robot_list) - 1
        else:
            if (self.allow_robot_name_duplicate):
                debug_print("robot name is duplicated. So change robot name")
                debug_print(f"change the robot name from {robot.name} to {robot.name + '_' + str(random.randint(0, 1000))}")
                self.robot_list.append(robot)
                self.robot_name_index[robot.name + "_" + str(random.randint(0, 1000))] = len(self.robot_list) - 1
                robot.name = robot.name + "_" + str(random.randint(0, 1000))
            else:
                raise ValueError("Robot name is duplicated")

    def get(self, name):
        return self.robot_list[self.robot_name_index[name]]
    
    def show_robot_info(self):
        for robot_obj in self.robot_list:
            robot_obj.robotinfo()
            
    def export_robot_list(self):
        return self.robot_list


class Robot(Agent):
    # task_list
    # {start_step : [destination, time]}

    def __init__(self, name="", velocity=[], path=[], position=[0, 0], personalspace=0, size = 10, mapsize=[100, 100], color="black", lidar_radius=10, mapinfo=None, task_list=[], destination=[0, 0], unique_num=-1):
        # should check the relationship of variable order
        super().__init__(name, velocity, path, position, personalspace,size,  color, {}, unique_num)

        self.destination = destination
        self.task_list = task_list
        self.isDoTask = True
        self.arrival_step = 0
        self.lidar_radius = lidar_radius

        # Settings Map Info
        self.ogm = OccupancyGridMap(np.ones((mapsize[0], mapsize[1])), 1)
        
        self.map_size = mapsize

        self.ogm.mapinfo = mapinfo

        self.lidar_offset = 0.02
        # self.lidar_offset = 0.1

        self.before_position = np.array(self.init_position)
        
        self.judge_area = True
        self.coverage_thre = 0.05

        # DWA settings(Local Planner)
        self.localgoal = None
        self.config = dwa.Config()
        self.config.robot_type = dwa.RobotType.rectangle
        self.yaw = math.pi * 9 / 8.0
        self.velocity = 0.0
        self.omega = 0.0
        self.x = np.array([self.position[0], self.position[1],
                    self.yaw, self.velocity, self.omega])

        if (mapinfo is not None):
            self.mapinfo = mapinfo
            self.location = list(map(lambda x : tuple(x), mapinfo))
            self.mapmask = np.ones(mapsize)
            for y in range(mapsize[1]):
                for x in range(mapsize[0]):
                    self.mapmask[y][x] = createMask([x, y], self.mapmask[y][x], self.location)

        self.stay_time_map = OccupancyGridMap(np.ones((mapsize[0], mapsize[1])), 1)
        
        
        self.is_reid = False
        
        # Store Variable
        self.log_store = ""
    
        self.globalgoal_trajectory = {}
        
        self.localgoal_trajectory = {}
        self.trajectory = {}
        
        self.count_people = {}
        self.count_people_by_step = {}
        
        self.count_people_num = 0
        
        
        
        
    def log(self, text, stdout_print=True):
        if (stdout_print):
            debug_print(text)
        self.log_store += f"{text}\n"

    def reset(self):
        self.position = list(self.init_position)
        self.path = self.init_path


    def dcircle(self, x, a, b):
        y1 = b + math.sqrt(1 - (x - a)*(x-a))
        y2 = b - math.sqrt(1 - (x - a)*(x-a))
        if (abs(y1 - self.position[1]) < abs(y2 - self.position[1])):
            return y1
        else:
            return y2


    def isArrival(self, dst = None, is_arrived=True):
        if (dst is None):
            dst = self.destination
        if (calcEuclidean(self.position, dst) < self.config.robot_radius):
            self.isArrived = True if is_arrived else False
            self.log("Arrived at destination")
            self.log(f"ego pos : {self.position}")
            self.log(f"")
            return True
        elif (len(self.path) == 0 and not is_projection_closer_to_a(self.before_position, dst, dst, self.position)):
            self.isArrived = True if is_arrived else False
            self.log("go past destination")
            self.log(f"bp : {self.before_position}")
            self.log(f"dst : {self.destination}")
            self.log(f"ego pos : {self.position}")
            self.log(f"path : {self.path}")
            self.log(f"is_projection : {is_projection_closer_to_a(self.before_position, dst, dst, self.position)}")
            return True
        # elif (np.sum(self.nm) / (self.nm.shape[0] * self.nm.shape[1]) < self.coverage_thre and self.judge_area):
        #     self.isArrived = True
        #     self.log(f"area_coverage : {np.sum(self.nm) / (self.nm.shape[0] * self.nm.shape[1])}")
        #     return True
        # return False
    def setTask(self, task_list):
        for task_idx in task_list:
            self.task_list[task_idx] = task_list[task_idx]

    def scan(self, agent_list):
        self.ogm.reset_data()
        _wall = makeWall(self.position, self.lidar_radius)
        # _wall = [[10, 10], [5, 5]]
        human_unique_num_dict = {
            agent.unique_num: agent for agent in agent_list}

        ogm_info = {"size": self.ogm.dim_cells}


        agent_value_list = [[agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info] for agent in agent_list]
        
        results = joblib.Parallel(n_jobs=get_availble_cpus(0.8), verbose=0, backend='threading')(joblib.delayed(one_azimuth_scan)(
            ogm_info, self.ogm.mapinfo, self.position, point, agent_value_list, self.lidar_radius) for point in _wall)


        time = np.array([0., 0., 0.])
        for result in results:
            self.ogm.data *= result[0]
            for _occulusion_agent in result[1]:
                if (_occulusion_agent not in self.count_people):
                    self.count_people[_occulusion_agent] = human_unique_num_dict[_occulusion_agent]
            time += result[2]
        self.ogm.data[int(self.position[1])][int(self.position[0])] = 0
        self.ogm.data *= self.mapmask
        return self.ogm.data, list(self.count_people.values())

  
    def setNonMeasureArea(self, nm):
        self.nm = nm
    def setOcculusion(self, occ):
        self.occ = occ
        
    def countPeople(self, step, agent_list, already_count_people_list,threshold=0.8):
        # occulusion_list = []
        # for agent in agent_list:
        #     try:
        #         if (self.ogm.get_data(agent.position) < threshold):
        #             occulusion_list.append(agent)
        #     except:
        #         pass
        count_people_list = {}
        for agent in agent_list:
            if (agent == [] or self.is_reid):
                pass
            else:
                if (step <= 1):
                    self.count_people_num += 1

                    count_people_list[agent.unique_num] = {
                        "agent": agent,
                        "position": agent.position[:],
                        "people_index": int(self.count_people_num)
                    }
                else:
                    # count_people_list_before_one_step = {i["agent"].unique_num: i for i in self.count_people_by_step[step - 1]}
                    if (agent.unique_num not in self.count_people_by_step[step - 1]):
                        self.count_people_num += 1
                        count_people_list[agent.unique_num] = {
                            "agent": agent,
                            "position": agent.position[:],
                            "people_index": int(self.count_people_num)
                        }
                    else:
                        count_people_list[agent.unique_num] = {
                            "agent": agent,
                            "position": agent.position[:],
                            "people_index": self.count_people_by_step[step - 1][agent.unique_num]["people_index"]
                        }
        self.count_people_by_step[step] = count_people_list

        # return

    def lidar_plot(self, robot_pos=[0, 0], mapinfo=None, agent_list=[], t=[], _wall=[], alpha=1, min_val=0, origin='lower'):
        """
        plot the grid map
        """

        fig = plt.figure(figsize=(20, 8))
        # plt.subplot(122)
        ax = fig.add_subplot(111)



        # Plot DATA
        ax.imshow(self.ogm.data, cmap="Purples", vmin=min_val,
                   vmax=1, origin=origin, interpolation='none', alpha=alpha)

        # Plot Agent
        for _agent in agent_list:
            oval = Ellipse(_agent.agent_info['rx'], _agent.agent_info['ry'], center=_agent.position)
            oval.rotate(_agent.agent_info['rad'], _agent.position)
            oval.draw(ax, display_focus=False, display_center=True)
        # oval.draw
        if (mapinfo is not None):
            ax.plot(mapinfo[0], mapinfo[1], lw=1.0, c="yellow")

        for i in _wall:
            ax.scatter(i[0], i[1], c="b", s=10)

        ax.scatter([i[0] for i in t], [i[1] for i in t], c="b", s=10)

        # Plot Robot
        ax.scatter(self.position[0], self.position[1],
                    c="y", edgecolor="orange", marker=",", s=20)

    
        ax.grid(True, which="minor", color="w", linewidth=.6, alpha=0.5)
        # plt.clim(0, 1)
        # cbar = fig.colorbar(ax = ax)
        # cbar.set_label("Value of Blind Spot")
        # plt.draw()
        plt.show()

    def move(self, agent_list=None):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

    def taskmove(self):
        pass

    def localplanner(self, agent_list = []):
        if (len(self.path) == 0):
            return False
        
        if (self.localgoal is None):
            try:
                self.localgoal = self.path[5][:]
                self.path = self.path[5:][:]
            except IndexError:
                self.localgoal = self.path[-1][:]
                self.path = []
        ob = np.array([agent.position for agent in agent_list])
        # x = np.array([self.position[0], self.position[1],
        #             self.yaw, self.velocity, self.omega])

        u, predicted_trajectory = dwa.dwa_control(
            self.x, self.config, self.localgoal, ob)
        self.x = dwa.motion(self.x, u, self.config.dt)
        # debug_print("after : {}".format(x))
        self.position = self.x[:2][:]
        # self.yaw = x[2]
        # self.velocity = x[3]
        # self.omega = x[4]

        dist_to_goal = math.hypot(x[0] - self.localgoal[0], x[1] - self.localgoal[1])
        if dist_to_goal <= self.config.robot_radius:
            self.localgoal = None


class AStarRobot(Robot):
    isTaskArrived = False

    def __init__(self, name="", velocity=[], path=[], position=[0, 0], personalspace=0, size = 10, mapsize=[100, 100], color="black", lidar_radius=50, mapinfo=None, task_list=[], unique_num=-1, destination=[0, 0], grid_size=2):
        super().__init__(name, velocity, path, position, personalspace, size,mapsize,
                         color, lidar_radius, mapinfo, task_list, destination, unique_num)

        self.grid_size = grid_size
        self.ox, self.oy = [], []
        self.task_dst, self.task_time_queue = [], []
        self.max_task_time = 100000
        self.yaw = math.pi / 8.0
        self.velocity = 0.0
        self.omega = 0.0


        # DWA settings
        self.config = dwa.Config()
        self.config.robot_type = dwa.RobotType.rectangle
        self.localgoal = None
        self.localgoal_list = []
        
        self.next_point_width = 5
        self.x = np.array([position[0], position[1],
                    self.yaw, self.velocity, self.omega])
        
        self.last_position = np.array(self.position)
        
        # 
        self.globalpath_list = []
        
    def robotinfo(self):
        table = PrettyTable()
        table.field_names = ["Property", "Value"]  
        
        table.add_row(["Robot name", self.name])
        table.add_row(["Grid size", self.grid_size])
        table.add_row(["ox", self.ox])
        table.add_row(["oy", self.oy])
        table.add_row(["Task dst", self.task_dst])
        table.add_row(["Task time queue", self.task_time_queue])
        table.add_row(["Max task time", self.max_task_time])
        table.add_row(["Yaw", self.yaw])
        table.add_row(["Velocity", self.velocity])
        table.add_row(["Omega", self.omega])
        table.add_row(["Config", self.config])
        
        debug_print("######### Robot Info ##########")
        debug_print(str(table))
        debug_print("###############################")
        
        
    def setDestination(self, dst: list):
        self.destination = dst

    def setObstacle(self, obstacles):
        for obstacle in obstacles:
            self.ox.append(obstacle[0])
            self.oy.append(obstacle[1])

    def globalplanner(self, dst=None, agent_list=[], pos=None):
        if (dst is None):
            if (len(self.task_dst) != 0):
                dst = self.task_dst[0]
            else:
                dst = self.destination
        if (dst[0] < self.grid_size and dst[1] < self.grid_size):
            dst = [self.grid_size, self.grid_size]
        # if (dst[0] == 0): dst[0] = self.grid_size
        # if (dst[1] == 0): dst[1] = self.grid_size
        a_star_instance, ox, oy = self._make_astar_instance(agent_list=agent_list)

        # debug_print("Calculate Path from {} to {}".format(self.position, dst))
        if (pos is None):
            pos = self.position[:]
        rx, ry, costmap = a_star_instance.planning(
            pos[0], pos[1], dst[0], dst[1])
        return list(zip(list(reversed(rx)), list(reversed(ry)))), costmap


    def localplanner(self, agent_list = []):
        
        if (self.localgoal is None):
            try:
                if (len(self.path) > self.next_point_width):
                    self.log(f"path : {self.path}")
                    self.localgoal = self.path[self.next_point_width][:]
                    self.path = self.path[self.next_point_width+1:][:]
                elif (len(self.path) == 1):
                    self.localgoal = self.path[0]
                    self.path = []                    
                elif (len(self.path) == 0):
                    self.log("no path")
                    self.localgoal = self.path[-1][:]
                    self.path = []
                else:
                    return False
                self.localgoal_list.append(self.localgoal)
                
            except IndexError:
                debug_print(f"path : {self.path}") 
                debug_print(f"localgoal : {self.localgoal}")  
        
        if self.localgoal is None:
            self.log("localgoal is None")
            # self.isArrived = True
            return
        
        agents_points = []
        for agent in agent_list:
            agent_points = generate_grid_points_in_ellipse(agent.agent_info['rx'], agent.agent_info['ry'], agent.position[0], agent.position[1], agent.agent_info['rad'])
            agents_points.extend(agent_points)
        ob = np.array(agents_points)
        try:
            u, predicted_trajectory = dwa.dwa_control(
                self.x, self.config, self.localgoal, ob)
        except IndexError:
            print(f"agent_list : {agent_list}")
            print(f"ob : {ob}")
            print(f"x : {self.x}")
            raise 
        self.x = dwa.motion(self.x, u, self.config.dt)
        self.velocity = self.x[3]
        self.yaw = self.x[2]
        
        self.last_position = np.array(self.position)
        self.position = np.array(self.x[:2])
        
        
        first_point = self.localgoal_list[-2][:] if len(self.localgoal_list) > 1 else self.before_position
        next_point = self.path[4][:] if len(self.path) > 5 else self.destination

        dist_to_localgoal = math.hypot(self.position[0] - self.localgoal[0], self.position[1] - self.localgoal[1])

        # Arrived at local goal
        if dist_to_localgoal <= self.config.robot_radius:
            self.log("Arrived at localgoal", stdout_print=False)
            self.before_position = self.localgoal
            self.localgoal = None
            return 
        
        # before 
        try:
            if self.localgoal is not None and not is_projection_closer_to_a(first_point, next_point, self.localgoal, self.position):
                self.log("Pass the destination")
                debug_print(f"first_point : {first_point}")
                debug_print(f"next_point : {next_point}")
                debug_print(f"self.localgoal : {self.localgoal}")
                debug_print(f"self.position : {self.position}")
                self.before_position = self.localgoal
                self.localgoal = None
        except Exception as e:
            debug_print(f"Error is {e}")
            debug_print(f"first_point : {first_point}")
            debug_print(f"next_point : {next_point}")
            debug_print(f"self.localgoal : {self.localgoal}")
            debug_print(f"self.position : {self.position}")
   

    # Args
    # task : [destination, time]
    def taskclient(self, start_time, task):
        if (len(self.task_list) == 0 and self.isDoTask and self.isArrived):
            if (task[1] == 0):
                task[1] = 1000
            debug_print("task_list : {}".format(task))
            self.task_list = {start_time : task}


    # not use area value (To be consistent with other classes.)
    def move(self, step, agent_list=[], dst=None, area=None):
        if (dst is None):
            dst = self.destination
            
        if (self.localgoal is None and len(self.path) == 0 and  not self.isArrived):
            debug_print("set global path")
            # temporal implement
            path = []
            pos = []
            tmp_path = []
            
            if (len(self.task_list) > 0):
                for idx, task in enumerate(self.task_list):
                    if (idx == 0):
                        pos = self.position[:]
                    tmp_path, _ = self.globalplanner(dst=task, agent_list=agent_list, pos=pos)
                    print(f"calculate path from : {pos} to : {task}")
                    pos = task[:]
                    path.extend(tmp_path)
                tmp_path = path[:]
            else:
                tmp_path, _ = self.globalplanner(dst = dst, agent_list=agent_list)
            # 
            if (len(tmp_path) > 1):
                self.path = tmp_path
                self.globalpath_list.append(list(tmp_path))
        # elif (self.localgoal is None and len(self.path) == 1):
        print(f"path : {self.path}, ego : {self.position}, dst : {self.destination}")
        repeat_count = 1
        for _ in range(repeat_count):
            if (not self.isArrived):
                self.localplanner(agent_list)
        
        self.globalgoal_trajectory[step] = np.array(self.destination)
        if (self.localgoal is None):
            print(f"local goal is None")
            self.localgoal_trajectory[step] = self.localgoal
        else:
            self.localgoal_trajectory[step] = np.array(self.localgoal)
        self.isArrival()
        # isAssignTask = False
        # if (dst is None):
        #     dst = self.destination

        # if (step in self.task_list):
        #     # if destination is the outside of map, it doesn't assign task
        #     if (not point_in_polygon(self.task_list[step][0], self.mapinfo)):
                
        #         debug_print("The destination of assigned is the outside of map")
        #         debug_print("It can't assign task")
        #         debug_print("destination : {}".format(self.task_list[step][0]))
        #         debug_print("reset task_list")
        #         self.task_list = {}
        #     else:
        #         if (len(self.task_dst) > 0):
        #             debug_print("Add Task to Queue")
        #             self.task_time_queue.append(self.task_list[step][1] + step)
        #         else:
        #             debug_print("Assign Task")
        #             debug_print("Destination : {}".format(self.task_list[step][0]))
        #             debug_print("Still {} step".format(
        #                 self.task_list[step][1] + step))

        #             self.max_task_time = self.task_list[step][1] + step
        #         self.task_dst.append(self.task_list[step][0])
        #         isAssignTask = True
        #         dst = self.task_list[step][0]
        # if (len(self.task_dst) != 0):
        #     if (self.max_task_time < step or self.isArrival(self.task_dst[0])):
        #         if (len(self.task_dst) == 1):
        #             debug_print("Reset Destination")
        #             self.task_dst = []
        #             self.task_time_queue = []
        #             self.task_list = {}
        #         elif (len(self.task_dst) > 1):
        #             debug_print("Assign Next Destination")
        #             self.task_dst = self.task_dst[1:]
        #             self.max_task_time = self.task_time_queue[0]
        #             self.task_time_queue = self.task_time_queue[1:]
        #             dst = self.task_dst
        #         self.path = []
        # # if (not isAssignTask and len(self.path) > 0):
        # #     self.position[0] = self.path[0][0]
        # #     self.position[1] = self.path[0][1]
        # #     self.path = self.path[1:]
        # #     return

        # # debug_print("Execute Local Planner...")
        # # self.localplanner(agent_list, dst)
        # if (len(self.path) == 0 or isAssignTask):
        #     # debug_print("Calculate Global Plan...")
        #     # debug_print("Destination : {}".format(dst))
        #     # debug_print("Robot Position : {}".format(self.position)

        #     if (calcEuclidean(self.position, dst) <= 2):
        #         self.path.append(self.position)
        #     else:
        #         tmp_path, costmap = self.globalplanner(dst=dst, agent_list=agent_list)
        #         self.globalpath_list.append(list(tmp_path))
        #         idx = 0
        #         self.path = [tmp_path[0]]
        #         for i in range(len(tmp_path)):
        #             # debug_print(self.path)
        #             # debug_print(tmp_path)
        #             # debug_print(self.path[idx])
        #             # debug_print(tmp_path[i])
        #             if (calcEuclidean(self.path[idx], tmp_path[i]) > np.linalg.norm(self.velocity)):
        #                 self.path.append(tmp_path[i])
        #                 idx += 1
                
        
        # self.position = self.path[0]
        # self.path = self.path[1:]

    def _make_astar_instance(self, agent_list=[]):
        tm = TimeMeasure(5)
        tm.setTitle("Make Star Instance")
        tm.checkpoint(1)
        ox, oy = list(self.ox), list(self.oy)
        tm.checkpoint(1)
        tm.checkpoint(2)
        for agent in agent_list:
            # if (agent.position[0] < 0 or agent.position[1] < 0 and not agent.isArrived):
            #     continue
            # wall = makeWall(agent.position, agent.personalspace)
            # sample_points = sample_rotated_ellipse_points(agent.position, agent.agent_info['rx'], agent.agent_info['ry'], agent.agent_info['rad'], num_points=300)
            sample_points = generate_grid_points_in_ellipse(agent.agent_info['rx'], agent.agent_info['ry'], agent.position[0], agent.position[1], agent.agent_info['rad'])
            for wall_point in sample_points:
                ox.append(wall_point[0])
                oy.append(wall_point[1])

        a_star_instance = a_star.AStarPlanner(
            ox, oy, self.grid_size, self.personalspace)

        
        return a_star_instance, ox, oy


class AllAStarRobot(AStarRobot):
    dst = None
    
    def move(self, step, agent_list=[], dst=None):

        isAssignTask = False
        if (dst is None and self.dst is None):
            self.dst = self.destination

        if (step in self.task_list):
            # if destination is the outside of map, it doesn't assign task
            if (not point_in_polygon(self.task_list[step][0], self.mapinfo)):
                
                debug_print("The destination of assigned is the outside of map")
                debug_print("It can't assign task")
                debug_print("destination : {}".format(self.task_list[step][0]))
                debug_print("reset task_list")
                self.task_list = {}
            else:
                if (len(self.task_dst) > 0):
                    debug_print("Add Task to Queue")
                    self.task_time_queue.append(self.task_list[step][1] + step)
                else:
                    debug_print("Assign Task")
                    debug_print("Destination : {}".format(self.task_list[step][0]))
                    debug_print("Still {} step".format(
                        self.task_list[step][1] + step))

                    self.max_task_time = self.task_list[step][1] + step
                self.task_dst.append(self.task_list[step][0])
                # isAssignTask = True
                self.dst = self.task_list[step][0]
        if (len(self.task_dst) != 0):
            if (self.max_task_time < step or self.isArrival(self.task_dst[0])):
                if (len(self.task_dst) == 1):
                    debug_print("Reset Destination")
                    self.task_dst = []
                    self.task_time_queue = []
                    self.task_list = {}
                    self.dst = None
                elif (len(self.task_dst) > 1):
                    debug_print("Assign Next Destination")
                    self.task_dst = self.task_dst[1:]
                    self.max_task_time = self.task_time_queue[0]
                    self.task_time_queue = self.task_time_queue[1:]
                    self.dst = self.task_dst
                self.path = []
        # if (not isAssignTask and len(self.path) > 0):
        #     self.position[0] = self.path[0][0]
        #     self.position[1] = self.path[0][1]
        #     self.path = self.path[1:]
        #     return

        # debug_print("Execute Local Planner...")
        # self.localplanner(agent_list, dst)
        # if (len(self.path) == 0 or isAssignTask):
        #     # debug_print("Calculate Global Plan...")
        #     # debug_print("Destination : {}".format(dst))
        #     # debug_print("Robot Position : {}".format(self.position)

        #     if (calcEuclidean(self.position, dst) <= 2):
        #         self.path.append(self.position)
        #     else:
        #         tmp_path = self.globalplanner(dst=dst, agent_list=agent_list)
        #         idx = 0
        #         self.path = [tmp_path[0]]
        #         for i in range(len(tmp_path)):
        #             if (calcEuclidean(self.path[idx], tmp_path[i]) > np.linalg.norm(self.velocity)):
        #                 self.path.append(tmp_path[i])
        #                 idx += 1
        if (self.isArrival(self.dst)):
            self.isArrived = True
        else:
            tmp_path = self.globalplanner(dst = dst, agent_list = agent_list)
            if (len(tmp_path) > 1):
                self.position = tmp_path[1]
        # self.path = self.path[1:]
class AStarwithBlindSpotRobot(AStarRobot):
    def __init__(self, name="", velocity=[], path=[], position=[0, 0], personalspace=0, size = 10, mapsize=[100, 100], color="black", lidar_radius=50, mapinfo=None, task_list=[], unique_num=-1, destination=[0, 0], grid_size=2):
        super().__init__(name, velocity, path, position, personalspace, size, mapsize, color,
                         lidar_radius, mapinfo, task_list, unique_num, destination, grid_size)

        self.grid_size = grid_size
        self.ox, self.oy = [], []
        self.task_dst, self.task_time_queue = [], []
        self.max_task_time = 100000
        self.diff_lidar = np.ones((size[0], size[1]))

    def move(self, step, agent_list=[], dst=None):

        if (dst is not None):
            self.planning(step, agent_list, dst)
            return

        a_star_instance = self._make_astar_instance(agent_list=agent_list)

        if (step in self.task_list):
            if (len(self.task_dst) > 0):
                debug_print("Add Task to Queue")
                self.task_time_queue.append(self.task_list[step][1] + step)
            else:
                debug_print("Assign Task")
                debug_print("Destination : {}".format(self.task_list[step][0]))
                debug_print("Still {} step".format(
                    self.task_list[step][1] + step))

                self.max_task_time = self.task_list[step][1] + step
            self.task_dst.append(self.task_list[step][0])

        if (self.max_task_time < step):
            if (len(self.task_dst) == 1):
                debug_print("Reset Destination")
                self.task_dst = []
                self.task_time_queue = []
            elif (len(self.task_dst) > 1):
                debug_print("Reset Destination")
                self.task_dst = self.task_dst[1:]
                self.max_task_time = self.task_time_queue[0]
                self.task_time_queue = self.task_time_queue[1:]

        if (len(self.task_list) == 0):
            _x, _y = a_star_instance.planning(
                self.position[0], self.position[1], self.destination[0], self.destination[1])
            # task_step = 200 - (len(_x))
            task = self._calc_task_by_blindspot(task_step=30)
            self.task_list[step + 10] = task
            debug_print("add task : {}".format(self.task_list))

        rx, ry = [], []
        if (len(self.task_dst) == 0):
            rx, ry = a_star_instance.planning(
                self.position[0], self.position[1], self.destination[0], self.destination[1])
        else:
            try:
                rx, ry = a_star_instance.planning(
                    self.position[0], self.position[1], self.task_dst[0][0], self.task_dst[0][1])
            except IndexError:
                debug_print(self.task_dst)
                raise IndexError()

        if (len(rx) > 1):
            self.position[0] = list(reversed(rx))[1]
        # else: self.position[0] += self.velocity[0] * random.choice([-1, 1])
        if (len(ry) > 1):
            self.position[1] = list(reversed(ry))[1]
        # else: self.position[1] += self.velocity[1] * random.choice([-1, 1])

        # prevent to be out of range
        if (self.position[0] < 0):
            self.velocity[0] *= -1
        if (self.position[1] < 0):
            self.velocity[1] *= -1

    def _make_tasklist(self, step, task_step):
        pass

    def _calc_task_by_blindspot(self, task_step, split_num=4):
        debug_print("calculate task by blindspot")
        size = self.ogm.data.shape
        center_list = np.array([[(int(size[0] / (split_num*2) + x * (size[0] / split_num)), int(size[1] / (
            split_num*2) + y * (size[1] / split_num))) for y in range(split_num)] for x in range(split_num)])
        blind_list = np.array([[0 for _ in range(split_num)]
                              for _ in range(split_num)])
        for y in range(size[1]):
            idx_y = int(y / (size[1] / split_num))
            for x in range(size[0]):
                idx_x = int(x / (size[0] / split_num))

                blind_list[idx_x][idx_y] += self.diff_lidar[x][y]

        idx = np.unravel_index(np.argmax(blind_list), blind_list.shape)
        try:
            task_list = [center_list[idx_x][idx_y], task_step]
        except IndexError:
            debug_print(center_list.shape)
            debug_print(blind_list.shape)
        return task_list

    def lidar(self, agent_list, lidar_radius=40):
        self.ogm.reset_data()
        occulusion_agent_list = []
        for map_point in self.ogm.mapinfo:
            laser_beams = lg.bresenham(self.position, map_point)

            occulusion_agent = [10000, None]
            isOcculusion_laser_beams = [False for i in range(len(laser_beams))]

            for agent in agent_list:
                checkOcculusion = [calcEuclidean(agent.position, laser_beam) < (
                    agent.personalspace) for laser_beam in laser_beams]
                if any(checkOcculusion):
                    isOcculusion_laser_beams = [x | y for x, y in zip(
                        isOcculusion_laser_beams, checkOcculusion)]
                    occ_idx = checkOcculusion.index(True)
                    if (occ_idx < occulusion_agent[0] and calcEuclidean(agent.position, self.position) < lidar_radius):
                        occulusion_agent[0] = occ_idx
                        occulusion_agent[1] = agent

            laser_point_list = laser_beams
            if any(isOcculusion_laser_beams):
                lb_idx = isOcculusion_laser_beams.index(True)
                laser_point_list = laser_beams[:lb_idx]

                if occulusion_agent[1] is not None and occulusion_agent[1] not in occulusion_agent_list:
                    occulusion_agent_list.append(occulusion_agent[1])

            for laser_beam in laser_point_list:
                if (calcEuclidean(laser_beam, self.position) > lidar_radius):
                    continue

                if (laser_beam.tolist() in self.ogm.mapinfo):
                    break
                self.ogm.set_data(laser_beam, 0)
        self.diff_lidar += ((self.ogm.data - 0.5) * 0.1)
        self.diff_lidar = np.minimum(np.maximum(self.diff_lidar, 0), 1)

        return self.ogm.data, occulusion_agent_list


class SweepRobot(AStarRobot):

    def __init__(self, name="", velocity=[], path=[], position=[0, 0], personalspace=0, size = 10, mapsize=[100, 100], color="black", lidar_radius=50, mapinfo=None, task_list=[], unique_num=-1, destination=[0, 0], grid_size=2, weight={}, bias={}):
        super().__init__(name, velocity, path, position, personalspace, size,mapsize,
                         color, lidar_radius, mapinfo, task_list, unique_num, destination, grid_size)
        
        self.resolution = 10
        self.judge_area = True
        self.coverage_thre = 0.05

        self.ox = [0.0, 0.0, 100.0, 100.0, 0.0]
        self.oy = [0.0, 100.0, 100.0, 0.0, 0.0]
        self.globalgoal_trajectory = {}
        self.localgoal_trajectory = {}

        self.globalpath_list = []
        self.before_position = np.array(self.init_position)
        
    def robotinfo(self):
        table = PrettyTable()
        table.field_names = ["Property", "Value"] 
        
        table.add_row(["Robot name", self.name])
        table.add_row(["Grid size", self.grid_size])
        table.add_row(["Task dst", self.task_dst])
        table.add_row(["Task time queue", self.task_time_queue])
        table.add_row(["Max task time", self.max_task_time])
        table.add_row(["Yaw", self.yaw])
        table.add_row(["Velocity", self.velocity])
        table.add_row(["Omega", self.omega])
        table.add_row(["Config", self.config])
        table.add_row(["Position", self.position])
        table.add_row(["Destination", self.destination])
        table.add_row(["resolution", self.resolution])
        
        debug_print("######### Robot Info ##########")
        debug_print(str(table))
        debug_print("###############################")
        
    def isArrival(self, dst = None, is_arrived=True):
        if (dst is None):
            dst = self.destination
        if (calcEuclidean(self.position, dst) < self.config.robot_radius):
            self.isArrived = True if is_arrived else False
            self.log("Arrived at destination")
            self.log(f"ego pos : {self.position}")
            self.log(f"")
            return True
        elif (len(self.path) == 0 and not is_projection_closer_to_a(self.before_position, dst, dst, self.position)):
            self.isArrived = True if is_arrived else False
            self.log("go past destination")
            self.log(f"bp : {self.before_position}")
            self.log(f"dst : {self.destination}")
            self.log(f"ego pos : {self.position}")
            return True
        elif (np.sum(self.nm) / (self.nm.shape[0] * self.nm.shape[1]) < self.coverage_thre and self.judge_area):
            self.isArrived = True
            self.log(f"area_coverage : {np.sum(self.nm) / (self.nm.shape[0] * self.nm.shape[1])}")
            return True
        return False
    def setPath(self, path):
        self.path = path
    def globalplanner(self, dst=None, agent_list=[], area = None):
        self.log(f"ox : {self.ox}")
        self.log(f"oy : {self.oy}")
        self.log(f"self.position : {self.position}")
        rx, ry = gbsweep_planning(self.ox, self.oy, self.resolution, start_position=[[-100, 0], self.position])
        new_path = list(zip(rx, ry))
        self.global_path_list.append(list(new_path))
        self.position = np.array(new_path[0])
        self.x[0] = self.position[0]
        self.x[1] = self.position[1]
        self.destination = np.array(new_path[-1])
        self.before_position = np.array(new_path[0])
        return new_path
    def move(self, step, agent_list=[], dst=None, area=None):
        if (dst is None):
            dst = self.destination
        if (self.localgoal is None and len(self.path) == 0 and not self.isArrived):
            self.log("set global path")
            tmp_path = self.globalplanner(dst=dst, agent_list=agent_list)
            if (len(tmp_path) > 1):
                self.path = tmp_path
                self.globalpath_list.append(list(tmp_path))
            self.log(str(self.path))
        for _ in range(self.next_point_width):
            if (not self.isArrived):
                self.localplanner(agent_list)
        self.globalgoal_trajectory[step] = np.array(self.destination)
        if (self.localgoal is None):
            self.localgoal_trajectory[step] = self.localgoal
        else:
            self.localgoal_trajectory[step] = np.array(self.localgoal)
        self.isArrival()

    def localplanner(self, agent_list = []):
        if (self.localgoal is None):
            try:
                if (len(self.path) > 1):
                    self.localgoal = self.path[0][:]
                    self.path = self.path[1:][:]
                elif (len(self.path) > 0):
                    self.localgoal = self.path[-1][:]
                    self.path = []
                else:
                    return False
                self.localgoal_list.append(self.localgoal)
            except IndexError:
                debug_print(f"path : {self.path}")
                debug_print(f"localgoal : {self.localgoal}")
        if self.localgoal is None:
            self.log("localgoal is None")
            self.isArrived = True
            return
        agents_points = []
        for agent in agent_list:
            agent_points = generate_grid_points_in_ellipse(agent.agent_info['rx'], agent.agent_info['ry'], agent.position[0], agent.position[1], agent.agent_info['rad'])
            agents_points.extend(agent_points)
        ob = np.array(agents_points)
        u, predicted_trajectory = dwa.dwa_control(
            self.x, self.config, self.localgoal, ob)
        self.x = dwa.motion(self.x, u, self.config.dt)
        self.velocity = self.x[3]
        self.yaw = self.x[2]
        
        self.last_position = np.array(self.position)
        self.position = np.array(self.x[:2])
        
        
        first_point = self.localgoal_list[-2][:] if len(self.localgoal_list) > 1 else self.before_position
        next_point = self.path[4][:] if len(self.path) > 5 else self.destination

        dist_to_localgoal = math.hypot(self.position[0] - self.localgoal[0], self.position[1] - self.localgoal[1])

        # Arrived at local goal
        if dist_to_localgoal <= self.config.robot_radius:
            self.log("Arrived at localgoal", stdout_print=False)
            self.before_position = self.localgoal
            self.localgoal = None
            return 
        
        # before 
        try:
            if not is_projection_closer_to_a(first_point, next_point, self.localgoal, self.position):
                self.log("Pass the destination")
                self.before_position = self.localgoal
                self.localgoal = None
        except:
            debug_print(f"first_point : {first_point}")
            debug_print(f"next_point : {next_point}")
            debug_print(f"self.localgoal : {self.localgoal}")
            debug_print(f"self.position : {self.position}")


class AStarBS(AStarRobot):
    count = -100
    def __init__(self, name="", velocity=[], path=[], position=[0, 0], personalspace=0, size = 10, mapsize=[100, 100], color="black", lidar_radius=50, mapinfo=None, task_list=[], unique_num=-1, destination=[0, 0], grid_size=2, weight={}, bias={}):
        super().__init__(name, velocity, path, position, personalspace, size,mapsize,
                         color, lidar_radius, mapinfo, task_list, unique_num, destination, grid_size)
        
        self.weight = weight
        self.bias = bias

    def robotinfo(self):
        table = PrettyTable()
        table.field_names = ["Property", "Value"] 
        
        table.add_row(["Robot name", self.name])
        table.add_row(["Grid size", self.grid_size])
        table.add_row(["Task dst", self.task_dst])
        table.add_row(["Task time queue", self.task_time_queue])
        table.add_row(["Max task time", self.max_task_time])
        table.add_row(["Yaw", self.yaw])
        table.add_row(["Velocity", self.velocity])
        table.add_row(["Omega", self.omega])
        table.add_row(["Config", self.config])
        table.add_row(["Weight", self.weight])
        table.add_row(["Bias", self.bias])
        table.add_row(["Position", self.position])
        table.add_row(["Destination", self.destination])
        
        debug_print("######### Robot Info ##########")
        debug_print(str(table))
        debug_print("###############################")
        
    def setBias(self, bias):
        self.bias = bias
        
    def setWeight(self, weight):
        self.weight = weight

    def move(self, step, agent_list=[], dst=None, area = None):

        if (dst is None):
            dst = self.destination
            
        if (len(self.path) == 0 and  not self.isArrived):
            self.log("set global path")
            
            tmp_path, _, _, _, _, _ = self.globalplanner(dst=dst, agent_list=agent_list, area=area)
            
            if (len(tmp_path) > 1):
                self.path = tmp_path
                self.globalpath_list.append(list(tmp_path))
            self.log(str(self.path))
        
        for _ in range(10):
            if (not self.isArrived):
                self.localplanner(agent_list)
        # debug_print(f"position : {self.position}")
        # debug_print(f"global path : {self.path}")
        # self.localplanner(agent_list)
        self.globalgoal_trajectory[step] = np.array(self.destination)
        if (self.localgoal is None):
            self.localgoal_trajectory[step] = self.localgoal
        else:
            self.localgoal_trajectory[step] = np.array(self.localgoal)

    def globalplanner(self, dst=None, agent_list=[], area = None):
        debug_print("Global Planner")
        if (dst is None):
            if (len(self.task_dst) != 0):
                dst = self.task_dst[0]
            else:
                dst = self.destination
        if (dst[0] < self.grid_size and dst[1] < self.grid_size):
            dst = [self.grid_size, self.grid_size]
        if (dst[0] == 0): dst[0] = self.grid_size
        if (dst[1] == 0): dst[1] = self.grid_size

        debug_print("Load AstarInsatnce")
        a_star_instance, ox, oy = self._make_astar_instance(agent_list=agent_list)

        debug_print("Calculate Path from {} to {}".format(self.position, dst))
        rx, ry, all_costmap, bs_costmap, time_costmap,org_costmap, bs_cache = a_star_instance.planning(
            self.position[0], self.position[1], dst[0], dst[1], agent_list, area, self.max_task_time)
        return list(zip(list(reversed(rx)), list(reversed(ry)))), all_costmap, bs_costmap, time_costmap,org_costmap, bs_cache
    
    def _make_astar_instance(self, agent_list=[]):
        ox, oy = list(self.ox), list(self.oy)
        for agent in agent_list:
            sample_points = generate_grid_points_in_ellipse(agent.agent_info['rx'], agent.agent_info['ry'], agent.position[0], agent.position[1], agent.agent_info['rad'])
            for wall_point in sample_points:
                ox.append(wall_point[0])
                oy.append(wall_point[1])

        lidar_info = {'area_size' : self.ogm.dim_cells, 'mapinfo': self.mapinfo, 'lidar_radius': self.lidar_radius, 'speed': self.config.max_speed}
        a_star_instance = AStarBSPlanner(
            ox, oy, self.grid_size, self.personalspace, lidar_info, width=self.map_size[0] , bias=self.bias, weight=self.weight)
        return a_star_instance, ox, oy

class WayPointRobot(AStarBS):
    def __init__(self, name="", velocity=[], path=[], position=[0, 0], personalspace=0, size = 10, mapsize=[100, 100], color="black", lidar_radius=50, mapinfo=None, task_list=[], unique_num=-1, destination=[0, 0], grid_size=2, weight={}, bias={}):
        super().__init__(name, velocity, path, position, personalspace, size,mapsize,
                         color, lidar_radius, mapinfo, task_list, unique_num, destination, grid_size)
        
        # mode 
        # 1 : 人流
        # 2 : 人数
        self.mode = 2
        self.always = False
        self.judge_area = True
        self.coverage_thre = 0.05
        
        self.ob_list = []
        
    def globalplanner(self, dst=None, agent_list=[], area = None, waypoints = None):
        self.log("Global Planner")
        if (waypoints is None):
            cost_map = create_demand_layer_map(70, 70)
            if (self.mode == 2):
                cost_map = self.ogm.data
            # waypoints = waypoint_generator(cost_map, sample_size = sample_size_decider(cost_map))
            waypoints = searchDsts(cost_map, None, 20)
            if (len(waypoints) > 20):
                idx_list = [random.randint(0, len(waypoints)) for i in range(15)]
                waypoints = [waypoints[idx] for idx in idx_list]
            
            waypoints.insert(0, [self.position[0], self.position[1]])
            
        # waypoints.append(self.destination)
        perm, dis = solver(waypoints)
        
        new_path = []
        for idx in perm:
            lg = LocalGoal(position=waypoints[idx], range=20)
            new_path.append(lg)
        self.log(new_path)
        
        previous_point = self.position
        
        # for perm_i in perm[1:]:
        #     a_star_instance, ox, oy = self._make_astar_instance(agent_list=agent_list)
        #     rx, ry, _ = a_star_instance.planning(
        #         previous_point[0], previous_point[1], waypoints[perm_i][0], waypoints[perm_i][1]
        #     )
        #     new_path.extend(list(zip(list(reversed(rx)), list(reversed(ry)))))
        #     previous_point = list(waypoints[perm_i])

        # Last Onemile
        # rx, ry, _ = a_star_instance.planning(
        #     previous_point[0], previous_point[1], self.position[0], self.position[1]
        # )
        # new_path.extend(list(zip(list(reversed(rx)), list(reversed(ry)))))

        self.destination = np.array(new_path[-1])
        # self.before_position = np.array(new_path[0])
        self.before_position = np.array(self.position)
        self.log(f"change destination : {self.destination}", stdout_print=False)
        return new_path, waypoints
    
    def isArrival(self, dst = None, is_arrived=True):
        if (dst is None):
            dst = self.destination
        if (calcEuclidean(self.position, dst) < self.config.robot_radius):
            self.isArrived = True if is_arrived else False
            self.log("Arrived at destination")
            self.log(f"ego pos : {self.position}")
            self.log(f"")
            return True
        elif (len(self.path) == 0 and not is_projection_closer_to_a(self.before_position, dst, dst, self.position)):
            self.isArrived = True if is_arrived else False
            self.log("go past destination")
            self.log(f"bp : {self.before_position}")
            self.log(f"dst : {self.destination}")
            self.log(f"ego pos : {self.position}")
            return True
        elif (np.sum(self.nm) / (self.nm.shape[0] * self.nm.shape[1]) < self.coverage_thre and self.judge_area):
            self.isArrived = True
            self.log(f"area_coverage : {np.sum(self.nm) / (self.nm.shape[0] * self.nm.shape[1])}")
            return True
        return False
    def _make_astar_instance(self, agent_list=[]):
        ox, oy = list(self.ox), list(self.oy)
        for agent in agent_list:
            sample_points = generate_grid_points_in_ellipse(agent.agent_info['rx'], agent.agent_info['ry'], agent.position[0], agent.position[1], agent.agent_info['rad'])
            for wall_point in sample_points:
                ox.append(wall_point[0])
                oy.append(wall_point[1])

        a_star_instance = a_star.AStarPlanner(
            ox, oy, self.grid_size, self.personalspace)

        return a_star_instance, ox, oy



    def localplanner(self, area, agent_list = []):
        if (self.always):
            flag = False
            new_waypoints = []
            for lg in self.path:
                if (lg.isClear(area = area, use_range = True)):
                    flag = True
                    self.localgoal = None
                else:
                    new_waypoints.append(lg)
        
        if (self.localgoal is None):
            try:
                # if (len(self.path) > 1):
                #     self.localgoal = self.path[4][:]
                #     self.path = self.path[5:][:]
                # elif (len(self.path) > 0):
                #     self.localgoal = self.path[-1][:]
                #     self.path = []
                # else:
                    # return False
                if (not self.always):
                    flag = False
                    new_waypoints = []
                    for lg in self.path:
                        if (lg.isClear(area = area, use_range = True)):
                            flag = True
                        else:
                            new_waypoints.append(lg)
                if (flag):
                    debug_print(f"flag on")
                    debug_print(f"path : {self.path}")
                    debug_print(f"new_waypoints : {new_waypoints}")
                    new_waypoints.insert(0, [self.position[0], self.position[1]])
                    # new_waypoints.append(self.destination)
                    new_path, _ = self.globalplanner(waypoints=new_waypoints)
                    self.path = new_path
                    self.globalpath_list.append(list(new_path))
                if (len(self.path) > 0):
                    self.localgoal = self.path[0]
                    self.path = self.path[1:]

                debug_print(f"path len : {len(self.path)}")
                debug_print(f"Localgoal is {self.localgoal}")
                self.localgoal_list.append(self.localgoal)
                
            except IndexError:
                debug_print(f"path : {self.path}") 
                debug_print(f"localgoal : {self.localgoal}") 
        
        if self.localgoal is None:
            self.log("localgoal is None")
            self.isArrived = True

            return
        
        agents_points = []
        for agent in agent_list:
            agent_points = generate_grid_points_in_ellipse(agent.agent_info['rx'], agent.agent_info['ry'], agent.position[0], agent.position[1], agent.agent_info['rad'])
            agents_points.extend(agent_points)
        agents_points.extend(self.mapinfo)
        # self.ob_list.append(agents_points)
        ob = np.array(agents_points)
        u, predicted_trajectory = dwa.dwa_control(
            self.x, self.config, self.localgoal, ob)
        self.x = dwa.motion(self.x, u, self.config.dt)
        self.velocity = self.x[3]
        self.yaw = self.x[2]
        
        self.last_position = np.array(self.position)
        self.position = np.array(self.x[:2])
        
        
        first_point = self.localgoal_list[-2] if len(self.localgoal_list) > 1 else self.before_position
        next_point = self.path[4] if len(self.path) > 5 else self.destination

        dist_to_localgoal = math.hypot(self.position[0] - self.localgoal[0], self.position[1] - self.localgoal[1])

        # Arrived at local goal
        if dist_to_localgoal <= self.config.robot_radius:
            self.log("Arrived at localgoal", stdout_print=False)
            self.before_position = self.localgoal
            self.localgoal = None
            return 
        # before 
        try:
            if not is_projection_closer_to_a(first_point, next_point, self.localgoal, self.position):
                self.log("Pass the destination")
                self.before_position = self.localgoal
                self.localgoal = None
        except:
            debug_print(f"first_point : {first_point}")
            debug_print(f"next_point : {next_point}")
            debug_print(f"self.localgoal : {self.localgoal}")
            debug_print(f"self.position : {self.position}")
   
    def move(self, step, agent_list=[], dst=None, area=None):
        if (dst is None):
            dst = self.destination
        # if (area is None):
        #     area = self.occ
            # debug_print("area is None")
        # if (step == 1):
        #     area = self.nm
        if (self.localgoal is None and len(self.path) == 0 and  not self.isArrived):
            self.log("set global path")
            tmp_path, _ = self.globalplanner(dst=dst, agent_list=agent_list)
            
            if (len(tmp_path) > 1):
                self.path = tmp_path
                self.globalpath_list.append(list(tmp_path))
            self.log(str([self.path]))
        
        for _ in range(self.next_point_width):
            if (not self.isArrived):
                self.localplanner(area=self.nm, agent_list = agent_list)
        self.globalgoal_trajectory[step] = np.array(self.destination)
        if (self.localgoal is None):
            self.localgoal_trajectory[step] = self.localgoal
        else:
            self.localgoal_trajectory[step] = np.array(self.localgoal)
        self.isArrival()
        
# class HumanFlowMeasure(AStarRobot):
#     def localplanner(self, agent_list = []):
        

class FrontierRobot(AStarBS):
    def __init__(self, name="", velocity=[], path=[], position=[0, 0], personalspace=0, size = 10, mapsize=[100, 100], color="black", lidar_radius=50, mapinfo=None, task_list=[], unique_num=-1, destination=[0, 0], grid_size=2, weight={}, bias={}):
        super().__init__(name, velocity, path, position, personalspace, size,mapsize,
                         color, lidar_radius, mapinfo, task_list, unique_num, destination, grid_size)

        self.weight = weight
        self.bias = bias

        # self.max_speed = 1.0  # 最大速度[m/s]
        # self.min_speed = -0.5  # 最小速度[m/s]
        # self.max_yaw_rate = 40.0 * math.pi / 180.0  # 最大回転速度[rad/s]
        # self.max_accel = 0.2  # 最大加速度[m/ss]
        # self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # 最大角加速度[rad/ss]
        # self.dt = 0.1  # タイムステップ[s]
        # self.predict_time = 3.0  # シミュレーション時間[s]
        # self.obstacle_radius = 0.5  # 障害物の安全距離[m]
        
        self.velocity = np.array([0, 0, 0])
        self.mass = 10
        self.position_dwa = np.array([*position, 0], dtype=np.float64)  # float64にキャスト
        self.velocity = np.array(self.velocity, dtype=np.float64)
 
    def move(self, step, agent_list=[], dst=None, area=None):
       dst = self.globalplanner(agent_list)
       self.log(f"dst : {dst}")
       self.localplanner(dst, agent_list)
       
    
    def globalplanner(self, agent_list=[], area=None):
        data = self.nm
        offset = np.array([0, 0])
        dst = np.array([0, 0])
        cnt = 0
        while (self.map_size[0] > 4*cnt):
            max_v = -1
            max_idx = [0, 0]
            max_child_grid = None
            n, m = data.shape
            for x_i in range(2):
                for y_i in range(2):
                    idx_y = y_i*n//2
                    idx_x = x_i*m//2
                    child_grid = data[idx_y:idx_y + m//2, idx_x:idx_x + n//2]
                    nm_v = np.sum(child_grid)
                    if (nm_v > max_v):
                        max_v = nm_v
                        max_idx = np.array([idx_x, idx_y])
                        max_child_grid = np.array(child_grid)
            offset += max_idx
            data = max_child_grid
            cnt += 1
            if (max_v > (n * m / 4) * 0.9):
                dst = offset + np.array([n, m]) // 4
                break
        return dst
        
    def localplanner(self, dst, agent_list=[]):
        dst_position = np.array(dst)
        
        agents_points = []
        for agent in agent_list:
            agent_points = generate_grid_points_in_ellipse(agent.agent_info['rx'], agent.agent_info['ry'], agent.position[0], agent.position[1], agent.agent_info['rad'])
            agents_points.extend(agent_points)
        agents_points.extend(self.mapinfo)
        
        vx, vy, yaw_rate = self.dwa_control(dst_position, np.array(agents_points))
        
        # 速度と位置の更新
        self.velocity[0] = vx  # X方向速度
        self.velocity[1] = vy  # Y方向速度
        self.velocity[2] = yaw_rate  # 回転速度
        self.log(f"velocity : {self.velocity}")
        self.position_dwa[:2] += np.array(self.velocity[:2]) * self.config.dt  # X, Y位置の更新
        self.position_dwa[2] += self.velocity[2] * self.config.dt  # 角度の更新
        self.position = self.position_dwa[:2]

    def dwa_control(self, goal, obstacles):
        """
        オムニホイール用のDWA (Dynamic Window Approach) の制御
        """
        min_cost = float('inf')
        best_vx, best_vy, best_yaw_rate = 0.0, 0.0, 0.0
        
        # 動的窓の計算
        velocities_x = np.linspace(self.config.min_speed, self.config.max_speed, num=10)
        velocities_y = np.linspace(self.config.min_speed, self.config.max_speed, num=10)
        yaw_rates = np.linspace(-self.config.max_yaw_rate, self.config.max_yaw_rate, num=10)
        
        for vx in velocities_x:
            for vy in velocities_y:
                for yaw_rate in yaw_rates:
                    trajectory = self.predict_trajectory(vx, vy, yaw_rate)
                    to_goal_cost = self.calc_to_goal_cost(trajectory, goal)
                    obstacle_cost = self.calc_obstacle_cost(trajectory, obstacles)
                    speed_cost = self.config.max_speed - np.linalg.norm([vx, vy])  # 速度のコスト
                    
                    total_cost = to_goal_cost + obstacle_cost + speed_cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_vx, best_vy, best_yaw_rate = vx, vy, yaw_rate
        
        return best_vx, best_vy, best_yaw_rate

    def predict_trajectory(self, vx, vy, yaw_rate):
        """
        現在の速度ベクトルで予測されるオムニホイールロボットの軌道
        """
        x, y, theta = self.position_dwa  # 現在の位置と姿勢
        trajectory = []
        time = 0.0
        while time <= self.config.predict_time:
            x += vx * math.cos(theta) - vy * math.sin(theta) * self.config.dt
            y += vx * math.sin(theta) + vy * math.cos(theta) * self.config.dt
            theta += yaw_rate * self.config.dt
            trajectory.append([x, y])
            time += self.config.dt
        
        return np.array(trajectory)

    def calc_to_goal_cost(self, trajectory, goal):
        """
        目標地点への距離を評価するコスト関数
        """
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        return math.hypot(dx, dy)

    def calc_obstacle_cost(self, trajectory, obstacles):
        """
        障害物との距離に基づくコスト計算
        """
        min_distance = float('inf')
        for obs in obstacles:
            for point in trajectory:
                dist = np.linalg.norm(point - obs)
                if dist < min_distance:
                    min_distance = dist
        
        if min_distance < self.config.obstacle_radius:
            return float('inf')  # 衝突する場合は無限大のコスト
        
        return 1.0 / min_distance  # 障害物が近いほどコストが高くなる
    
class AStarBSTask(AStarRobot):
    count = -100
    def __init__(self, name="", velocity=[], path=[], position=[0, 0], personalspace=0, size = 10, mapsize=[100, 100], color="black", lidar_radius=50, mapinfo=None, task_list=[], unique_num=-1, destination=[0, 0], grid_size=2, weight={}, bias={}):
        super().__init__(name, velocity, path, position, personalspace, size,mapsize,
                         color, lidar_radius, mapinfo, task_list, unique_num, destination, grid_size)
        
        self.weight = weight
        self.bias = bias
        
        debug_print(f"weight : {weight}")
        debug_print(f"bias : {bias}")
        
    def setBias(self, bias):
        self.bias = bias
        
    def setWeight(self, weight):
        self.weight = weight

    def move(self, step, agent_list=[], dst=None, area = None):

        isAssignTask = False
        if (dst is None):
            dst = self.destination

        if (step in self.task_list):
            # if destination is the outside of map, it doesn't assign task
            if (not point_in_polygon(self.task_list[step][0], self.mapinfo)):
                
                debug_print("The destination of assigned is the outside of map")
                debug_print("It can't assign task")
                debug_print("destination : {}".format(self.task_list[step][0]))
                debug_print("reset task_list")
                self.task_list = {}
            else:
                if (len(self.task_dst) > 0):
                    debug_print("Add Task to Queue")
                    self.task_time_queue.append(self.task_list[step][1] + step)
                else:
                    debug_print("Assign Task")
                    debug_print("Destination : {}".format(self.task_list[step][0]))
                    debug_print("Still {} step".format(
                        self.task_list[step][1] + step))

                    self.max_task_time = self.task_list[step][1] + step
                self.task_dst.append(self.task_list[step][0])
                isAssignTask = True
                dst = self.task_list[step][0]
        if (len(self.task_dst) != 0):
            if (self.max_task_time < step or self.isArrival(self.task_dst[0])):
                if (len(self.task_dst) == 1):
                    debug_print("Reset Destination")
                    self.task_dst = []
                    self.task_time_queue = []
                    self.task_list = {}
                elif (len(self.task_dst) > 1):
                    debug_print("Assign Next Destination")
                    self.task_dst = self.task_dst[1:]
                    self.max_task_time = self.task_time_queue[0]
                    self.task_time_queue = self.task_time_queue[1:]
                    dst = self.task_dst
                self.path = []
        if (len(self.path) == 0 or isAssignTask and self.isArrived):
            if (calcEuclidean(self.position, dst) <= 2):
                self.path.append(self.position)
            else:
                tmp_path = self.globalplanner(dst=dst, agent_list=agent_list, area = area)
                idx = 0
                self.path = [tmp_path[0]]
                for i in range(len(tmp_path)):
                    if (calcEuclidean(self.path[idx], tmp_path[i]) > np.linalg.norm(self.velocity)):
                        self.path.append(tmp_path[i])
                        idx += 1
        tmp_path = None
        if (not isAssignTask and self.count > 10 or len(self.path) == 0):
            tmp_path = self.globalplanner(dst=dst, agent_list=agent_list, area = area)
            self.count = 0
        else:
            self.count +=1
        if (tmp_path is None):
            self.position = self.path[0]
            self.path = self.path[1:]
        elif (len(tmp_path) > 1):
            self.position = tmp_path[0]
            self.path = tmp_path[1:]

    def globalplanner(self, dst=None, agent_list=[], area = None):
        debug_print("Global Planner")
        if (dst is None):
            if (len(self.task_dst) != 0):
                dst = self.task_dst[0]
            else:
                dst = self.destination
        if (dst[0] < self.grid_size and dst[1] < self.grid_size):
            dst = [self.grid_size, self.grid_size]
        if (dst[0] == 0): dst[0] = self.grid_size
        if (dst[1] == 0): dst[1] = self.grid_size

        debug_print("Load AstarInsatnce")
        a_star_instance, ox, oy = self._make_astar_instance(agent_list=agent_list)

        debug_print("Calculate Path from {} to {}".format(self.position, dst))
        rx, ry, all_costmap, bs_costmap, time_costmap,org_costmap, bs_cache = a_star_instance.planning(
            self.position[0], self.position[1], dst[0], dst[1], agent_list, area, self.max_task_time)
        return list(zip(list(reversed(rx)), list(reversed(ry)))), all_costmap, bs_costmap, time_costmap,org_costmap, bs_cache
    
    def _make_astar_instance(self, agent_list=[]):
        ox, oy = list(self.ox), list(self.oy)
        for agent in agent_list:
            sample_points = generate_grid_points_in_ellipse(agent.agent_info['rx'], agent.agent_info['ry'], agent.position[0], agent.position[1], agent.agent_info['rad'])
            for wall_point in sample_points:
                ox.append(wall_point[0])
                oy.append(wall_point[1])

        lidar_info = {'area_size' : self.ogm.dim_cells, 'mapinfo': self.mapinfo, 'lidar_radius': self.lidar_radius, 'speed': self.config.max_speed}
        a_star_instance = AStarBSPlanner(
            ox, oy, self.grid_size, self.personalspace, lidar_info, self.bias, self.weight)
        return a_star_instance, ox, oy

class CheckLocalPlanner(AStarRobot):
    localgoal = None
    
    def __init__(self, name="", velocity=[], path=[], position=[0, 0], personalspace=0, size = 10, mapsize=[100, 100], color="black", lidar_radius=50, mapinfo=None, task_list=[], unique_num=-1, destination=[0, 0], grid_size=2):
        super().__init__(name, velocity, path, position, personalspace, size,mapsize,
                         color, lidar_radius, mapinfo, task_list, unique_num, destination, grid_size)
        self.yaw = math.pi / 8.0
        self.velocity = 0.0
        self.omega = 0.0


        # DWA settings
        self.config = dwa.Config()
        self.config.robot_type = dwa.RobotType.rectangle
        
        self.config.max_speed = 2.0
        
        self.localgoal_list = []
        
        
    def move(self, step, agent_list = [], dst = None):
        
        if (self.isArrived):
            return False
        
        if (len(self.path) == 0):
            debug_print("set global path")
            self.setPath(dst=dst, agent_list=agent_list)

        self.localplanner(agent_list)


    def localplanner(self, agent_list = []):
        
        if (self.localgoal is None):
            try:
                self.localgoal = self.path[10][:]
                self.path = self.path[10:][:]
            except IndexError:
                self.localgoal = self.path[-1][:]
                self.path = []
                
            # Collision check
            if (len(self.localgoal_list) != 0):
                try:
                    if is_agent_between_points(self.localgoal_list[-1], self.localgoal, agent_list, self.config.robot_radius): 
                        debug_print("check obstacle\nchange local goal")
                        self.localgoal = self.path[10][:]
                        self.path = self.path[10:][:]
                except TypeError:
                    debug_print(f"before local goal : {self.localgoal_list[-1]}")
                    debug_print(f"now local goal : {self.localgoal}")
                    debug_print(f"robot radius : {self.config.robot_radius}")
                    
                
            self.localgoal_list.append(self.localgoal)
        agents_points = []
        for agent in agent_list:
            agent_points = generate_grid_points_in_ellipse(agent.agent_info['rx'], agent.agent_info['ry'], agent.position[0], agent.position[1], agent.agent_info['rad'])
            agents_points.extend(agent_points)
        ob = np.array(agents_points)
        # x = np.array([self.position[0], self.position[1],
                    # self.yaw, self.velocity, self.omega])
        u, predicted_trajectory = dwa.dwa_control(
            self.x, self.config, self.localgoal, ob)
        self.x = dwa.motion(self.x, u, self.config.dt)
        # debug_print("after : {}".format(x))
        self.position = self.x[:2][:]
        self.isStarted = True
        dist_to_localgoal = math.hypot(self.position[0] - self.localgoal[0], self.position[1] - self.localgoal[1])
        if dist_to_localgoal <= self.config.robot_radius:
            self.localgoal = None
        
        
            # self.isArrived = True


    def setPath(self, dst = None, agent_list = []):
        if (dst is None):
            dst = self.destination
        tmp_path = self.globalplanner(dst=dst, agent_list=agent_list)

        self.path = tmp_path
       


class ASLPwithHM(AStarRobot):
    def isOcculusion(self, top_left, bottom_right, threshold = 0.7, areaPCD=None):
        if (areaPCD is None):
            areaPCD = self.ogm.data
        x1, y1 = top_left
        x2, y2 = bottom_right
        # 矩形内の部分を抽出
        rectangle = areaPCD[y1:y2 + 1, x1:x2 + 1]
        size = rectangle.shape
        total_sum = 0
        for y in range(size[1]):
            for x in range(size[0]):
                circle = {'center' : [self.position[0], self.position[1]], 'radius' : self.lidar_radius}
                rectangle_dic = {'origin' : (top_left + bottom_right) / 2, 'size' : np.abs(top_left - bottom_right)}
                if (circleRectangleCollision(circle, rectangle_dic)):
                    total_sum += rectangle[y][x]
        if (total_sum / (size[0] * size[1]) > threshold):
            return True
        return False

    def isHumanMove(self, agent_list, human_trajectory):
        pass

    def checkHumanMove(self, agent_list, top_left, bottom_right, areaPCD=None):
        occulusion_agent = []
        if (self.isOcculusion(top_left=top_left, bottom_right=bottom_right)):
            for agent in agent_list:
                if (is_point_inside_rectangle(agent.position[0], agent.position[1], top_left, bottom_right)):
                    occulusion_agent.append(agent)
        
        if self.isHumanmove():
            pass

    def localplanner(self, agent_list = []):
        if (len(self.path) == 0):
            return False
        
        if (self.localgoal is None):
            debug_print("local goal sample : 10")
            try:
                self.localgoal = self.path[10][:]
                self.path = self.path[10:][:]
            except IndexError:
                self.localgoal = self.path[-1][:]
                self.path = []
                
        
        ob = np.array([agent.position for agent in agent_list])
        x = np.array([self.position[0], self.position[1],
                    self.yaw, self.velocity, self.omega])

        u, predicted_trajectory = dwa.dwa_control(
            x, self.config, self.localgoal, ob)
        x = dwa.motion(x, u, self.config.dt)
        # debug_print("after : {}".format(x))
        self.position = x[:2]
        self.yaw = x[2]
        self.velocity = x[3]
        self.omega = x[4]

        dist_to_goal = math.hypot(x[0] - self.localgoal[0], x[1] - self.localgoal[1])
        if dist_to_goal <= self.config.robot_radius:
            self.localgoal = None

    def move(self, step, agent_list = [], dst = None):
        self.localplanner(agent_list)

# class ASLPHM(AStarRobot):
    
    # def isOcculusion(self, top_left, bottom_right, threshold = 0.7, areaPCD=None):
    #     if (areaPCD is None):
    #         areaPCD = self.ogm.data
    #     x1, y1 = top_left
    #     x2, y2 = bottom_right

    #     # 矩形内の部分を抽出
    #     rectangle = areaPCD[x1:x2 + 1, y1:y2 + 1]
    #     size = rectangle.shape
    #     total_sum = 0
    #     for y in range(size[1]):
    #         for x in range(size[0]):
    #             circle = {'center' : [self.position[0], self.position[1]], 'radius' : self.lidar_radius}
    #             rectangle = {'origin' : (top_left + bottom_right) / 2, 'size' : np.abs(top_left - bottom_right)}
    #             if (circleRectangleCollision(circle, rectangle)):
    #                 total_sum += rectangle[x][y]

    #     if (total_sum / (size[0] * size[1]) > threshold):
    #         return True
    #     return False


    # def checkHumanMove(self, area, areaPCD=None):
    #     pass

    # def localplanner(self, agent_list = []):
    #     if (len(self.path) == 0):
    #         return False
        
    #     if (self.localgoal is None):
    #         try:
    #             self.localgoal = self.path[5][:]
    #             self.path = self.path[5:][:]
    #         except IndexError:
    #             self.localgoal = self.path[-1][:]
    #             self.path = []
    #     ob = np.array([agent.position for agent in agent_list])
    #     x = np.array([self.position[0], self.position[1],
    #                 self.yaw, self.velocity, self.omega])

    #     u, predicted_trajectory = dwa.dwa_control(
    #         x, self.config, self.localgoal, ob)
    #     x = dwa.motion(x, u, self.config.dt)
    #     # debug_print("after : {}".format(x))
    #     self.position = x[:2]
    #     self.yaw = x[2]
    #     self.velocity = x[3]
    #     self.omega = x[4]

    #     dist_to_goal = math.hypot(x[0] - self.localgoal[0], x[1] - self.localgoal[1])
    #     if dist_to_goal <= self.config.robot_radius:
    #         self.localgoal = None

    # def move(self, step, agent_list = [], dst = None):
    #     self.localplanner(agent_list)

