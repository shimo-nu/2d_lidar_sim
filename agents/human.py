import os, sys

import random
import datetime
import numpy as np
import pickle

from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from typing import List, Dict
from dataclasses import dataclass

from utils import generate_pseudo_trajectory, calculate_smooth_angle, makeHumanPath
from agents.agent import Agent
from debug import debug_print

class HumanManager():
    
    # Basic Agent (No move & Random Walk & RVO2)
    # org 後で消す
    # human_list = None
    # if (args.agent_mode == 'RVO2' or args.agent_mode == 'nomove'):
    #     pattern_num = 1
    #     agent_objects_by_pattern = dict()
    #     for pattern_name, agent_positions in agent_positions_by_pattern.items():
    #         agent_objects = []
    #         agent_pass_list = makeHumanPath(len(agent_positions), settings.steps + 1, wall, pos_list_args=agent_positions, radius=10, neighborDist=20)
    #         agent_pass_by_pattern[pattern_name] = agent_pass_list
    #         # else:
    #         #     agent_pass_list = agent_pass_by_pattern[pattern]
    #         #     agent_pass_list
    #         agent_keys = list(agent_pass_list.keys())
    #         print(agent_keys)
    #         for idx, agent_position in enumerate(agent_positions):
    #             yaw = 0
    #             if (len(agent_position) == 3):
    #                 yaw = agent_position[2]
    #             ## RVO2Human
    #             # agent = RVO2Human(name="smagv" + str(idx) + pattern, position=agent_position[:2], personalspace=10, size = 5,agent_info={"rx": 3, "ry": 1.5, "rad":yaw}, unique_num=int("{}{}".format(pattern_num, idx)))
    #             # agent = Agent(name="smagv" + str(idx) + pattern_name, position=agent_position[:2], personalspace=10, size = 5,agent_info={"rx":3, "ry": 1.5, "rad":yaw}, unique_num=int("{}{}".format(pattern_num, idx)))
    #             agent = RVO2Human(name="smagv" + str(idx) + pattern, position=agent_position, personalspace=10, size = 5,agent_info={"rx":3, "ry": 1.5, "rad":yaw}, unique_num=int("{}{}".format(pattern_num, idx)))
    #             agent.setPath(agent_pass_list[agent_keys[idx]])


    #             agent.isStarted = True
    #             agent_objects.append(agent)
    #         agent_objects_by_pattern[pattern_name] = agent_objects
    #         pattern_num += 1
    #     human_list = agent_objects_by_pattern[pattern]
    # elif (args.agent_mode == 'centrair'):
    #     # Centrair Agent
    #     human_list = []
    #     console.log("[Preparing] Initiating Human Instance...")
    #     with open(os.path.join(DATADIR, "master.centrair.pkl"), 'rb') as f:
    #         human_dict = pickle.load(f)
    #     # min_time = min([human_dict[human_key].loc_history[0][0]
    #     #                for human_key in human_dict])
    #     for idx, human_key in enumerate(human_dict):
    #         # if (idx > settings.people_maxnum):
    #         #     break
    #         if (len(human_list) > settings.people_maxnum):
    #             break
    #         converted_path_by_affinematrix = convertPathWithAffineMatrix(human_dict[human_key].loc_history, settings.affine_matrix)
    #         try:
    #             converted_path = deleteDuplicatePoint(converted_path_by_affinematrix)
    #             _human = CentrairHuman(name=human_dict[human_key].id, path=converted_path, personalspace=10, color="C" + str(idx + 1), affine_matrix=settings.affine_matrix, agent_info={"rx": 0.5, "ry": 0.3, "rad":np.pi/4}, unique_num=human_key)
    #             human_list.append(_human)
    #         except Exception as e:
    #             print(e)
    #             pass
    #     min_time = min([human.path[0][0]
    #            for human in human_list])
    #     for human in human_list:
    #         human.setStartTime(min_time)
    def __init__(self):
        self.human_list = []
        self.human_name_index = {}
        
    def __len__(self):
        return len(self.human_list)
    
    def __getitem__(self, key):
        if (key > len(self.human_list) - 1):
            raise IndexError("Index out of range")
        return self.human_list[key]
    
    def get(self, name):
        return self.human_list[self.human_name_index[name]]
        
    def loadPickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.human_list = pickle.load(f)

    def nomove(self, agent_positions):
        for idx, agent_position in enumerate(agent_positions):
            try:
                agent = Agent(name="smagv" + str(idx), position=agent_position[:2], personalspace=10, size = 5,agent_info={"rx":3, "ry": 1.5, "rad":agent_position[2]}, unique_num=int("{}".format(idx)))
                agent.isStarted = True
                self.human_list.append(agent)
                self.human_name_index["smagv" + str(idx)] = len(self.human_list) - 1
            except IndexError:
                debug_print("IndexError : so we skip this agent")
                debug_print(agent_position)

    def rvo2move(self, agent_positions, wall, steps, pattern, pattern_num):
        agent_pass_list = makeHumanPath(len(agent_positions), steps + 1, wall, pos_list_args=agent_positions, radius=10, neighborDist=20)
        agent_keys = list(agent_pass_list.keys())
        for idx, agent_position in enumerate(agent_positions):
            yaw = 0
            if (len(agent_position) == 3):
                yaw = agent_position[2]
            
            agent = RVO2Human(name="smagv" + str(idx) + pattern, position=agent_position, personalspace=10, size = 5,agent_info={"rx":3, "ry": 1.5, "rad":yaw}, unique_num=int("{}{}".format(pattern_num, idx)))
            agent.setPath(agent_pass_list[agent_keys[idx]])
            self.human_list[agent_keys[idx]] = agent
            
    
    def centrairmove():
        pass

class RandomHuman(Agent):
    wait_probability = 0.8
    # def __init__(self, **args):
    #     super().__init__(args)

    def setWaitProbability(self, wait_probability):
        self.wait_probability = wait_probability

    def setPath(self, seed):
        self.path = []
        self.directions = []
        self.seed = seed
        grid_size = 100
        meters_per_grid = 1.0
        wall_avoidance_prob = 0.2
        target_coordinates = (random.randrange(50, 100), random.randrange(50, 100))
        x_data, y_data, directions = generate_pseudo_trajectory(grid_size, meters_per_grid,current_position = self.position, wall_avoidance_prob=wall_avoidance_prob, target_coords=target_coordinates, random_seed=seed)

        for i in range(len(x_data)):
            self.path.append([x_data[i], y_data[i]])
        self.directions = directions
        tmp_path = self.path[:]
        self.agent_info['rad'] = self.directions[0]
        self.directions = self.directions[1:]
        self.position = self.path[0]
        self.path = self.path[1:]

        return tmp_path


    def move(self):
        self.position = self.path[0]
        self.path = self.path[1:]
        self.agent_info['rad'] = self.directions[0]
        self.directions = self.directions[1:]


class RandomSeedHuman(RandomHuman):
    cnt = 0

    def __init__(self, name="test", velocity="0", path=[], position=[], personalspace=0, color="k", unique_num=-1, wait_prob_list=[], velocity_ratio=[]):
        super().__init__(name, velocity, path, position, personalspace, color, unique_num)

        self.wait_prob_list = wait_prob_list
        self.velocity_ratio = velocity_ratio

    def reset(self):
        self.position = list(self.init_position)
        self.path = self.init_path
        self.cnt = 0

    def setCount(self, cnt):
        self.cnt = cnt

    def move(self):
        try:
            if (self.wait_prob_list[self.cnt] < self.wait_probability):
                self.position[0] += self.velocity[0] * \
                    self.velocity_ratio[self.cnt][0]
                self.position[1] += self.velocity[1] * \
                    self.velocity_ratio[self.cnt][1]
        except IndexError:
            raise IndexError("weird count")
        self.cnt += 1


class RVO2Human(Agent):

    def setPath(self, path):
        self.path = path
        self.directions = calculate_smooth_angle(path)

        self.agent_info['rad'] = self.directions[0]
        self.directions = self.directions[1:]
        self.position = self.path[0]
        self.path = self.path[1:]

        return path


    def move(self):
        idx = 0
        # self.dt = 0.1 # temp variable (dt is written at Parent Class)
        for point in self.path[1:]:
            next_distance = np.linalg.norm(np.array(self.position) - np.array(point))
            if (next_distance >= self.velocity * 0.1):
                break
            idx += 1
        try:
            self.position = self.path[idx]
            self.path = self.path[idx+1:]
            self.agent_info['rad'] = self.directions[idx]
            self.directions = self.directions[idx+1:]
        except IndexError:
            # print(f"idx ; {idx}")
            # print(f"path len : {len(self.path)}")
            # print(f"directions len : {len(self.directions)}")
            pass

class MapBasedHuman(Agent):

    def move(self):
        pass


@dataclass
class CentrairBaseHuman:
    id: int
    first_t: float
    first_x: float
    first_y: float
    last_t: float
    last_x: float
    last_y: float
    loc_history: List[List[float]]
    path_type: int  # 0: 両端が検出エリア外，1: firstが検出エリア内，2: lastが検出エリア内，3: first-lastが同じ検出エリア内，4: first-lastが異なる検出エリア内
    con_candidates: Dict[int, float]

    def __repr__(self) -> str:
        first_t = datetime.datetime.fromtimestamp(self.first_t)
        last_t = datetime.datetime.fromtimestamp(self.last_t)
        return '{}: {} @ ({}, {}) => {} @ ({}, {})'.format(self.id, first_t.strftime('%Y/%m/%d %H:%M:%S.%f'), self.first_x, self.first_y, last_t.strftime('%Y/%m/%d %H:%M:%S.%f'), self.last_x, self.last_y)


class CentrairHuman(Agent):
    cnt = 0
    time_span = 0.05

    def __init__(self, name="test", velocity="0", path=[], position=[], personalspace=0, size = 10, color="k",agent_info = {}, affine_matrix=[0, 0],  unique_num=-1):
        super().__init__(name, velocity, path, position, personalspace, size,  color, agent_info,unique_num)
        # if (len(position) == 0 and len(path) != 0):
        #     self.position = np.array([int(Decimal(str(self.path[self.cnt][1]/100)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)), int(
        #         Decimal(str(self.path[self.cnt][2]/100)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))]) + np.array(affine_matrix)
        self.position = np.array(path[0][1:])
        self.init_position = np.array(path[0][1:])
        self.cnt = 0
        self.affine_matrix = np.array(affine_matrix)
        self.init_path = np.array(path)

    def reset(self):
        self.position = list(self.init_position)
        self.path = self.init_path
        self.cnt = 0
        self.isArrived = False
        self.isStarted = False

    def setTimeSpan(self, _time_span):
        self.time_span = _time_span

    def setStartTime(self, time):
        self.st_time = round(time)

    def setCount(self, cnt):
        self.cnt = cnt

#     1step = 1min
    def move(self):

        if (self.isArrived):
            return 
        if (self.st_time < self.path[self.cnt][0]):
            self.st_time += 1
            return
        self.isStarted = True

        self.position = self.path[self.cnt][1:]
        self.cnt += 1
        # self.st_time += 1


        if (self.cnt+1 >= len(self.path)):
            self.isArrived = True

#     before movement
    # def move(self):
    #     if (self.st_time < round(self.path[self.cnt][0])):
    #         self.st_time += 1
    #         return
    #     self.isStarted = True
    #     while (self.st_time > round(self.path[self.cnt][0])):
    #         if (self.cnt+1 >= len(self.path)):
    #             self.isArrived = True
    #             break

    #         self.cnt += 1

    #     next_pos = [int(Decimal(str(self.path[self.cnt][1]/100)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)),
    #                 int(Decimal(str(self.path[self.cnt][2]/100)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))]
    #     self.position = (np.array(next_pos) + self.affine_matrix)
    #     self.st_time += 1

    def move_old(self):
        st_time = self.path[self.cnt][0]
        try:
            while (self.path[self.cnt + 1][0] - st_time < 1):
                self.cnt += 1
            next_pos = [int(Decimal(str(self.path[self.cnt][1]/100)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)),
                        int(Decimal(str(self.path[self.cnt][2]/100)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))]
            self.position = (np.array(next_pos) + self.affine_matrix)
        except IndexError:
            # self.cnt = 0
            return
        # if (self.path[self.cnt][0] - self.path[self.cnt - 1][0] >= self.time_span):
        # self.position = next_pos
        # self.cnt += 10

