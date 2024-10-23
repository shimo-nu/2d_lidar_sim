import os, sys
sys.path.append(os.pardir)

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
import rvo2

class HumanManager():
    def __init__(self):
        self.human_list = []
        self.human_name_index = {}
        
        self.mode = ''
        
    def __len__(self):
        return len(self.human_list)
    
    def __getitem__(self, key):
        if (key > len(self.human_list) - 1):
            raise IndexError("Index out of range")
        return self.human_list[key]
    
    def get(self, name):
        return self.human_list[self.human_name_index[name]]
        
    def loadpickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.human_list = pickle.load(f)
            
    def move(self, step):
        human_pose_list = {}
        for human in self.human_list:
            human_pose = []
            if (not human.isArrived and human.isStarted):
                human_pose = list(human.position)
            human.move()
            human_pose_list[human.unique_num] = human_pose
            
        if (self.mode == 'RVO2'):
            self.sim.doStep()
        return human_pose_list
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

    def rvo2move(self, step, r):
        human_pose_list = {}
        for human in self.human_list:
            human_pose = []
            if (not human.is_arrived and human.is_started):
                human_pose = list(human.position)
                human_pose.append(float(human.agent_info['rad']))
            human.move()
            human_pose_list[human.unique_num] = human_pose
        self.sim.doStep()
        return human_pose_list
    
    def fixpathsetup(self, agent_path_list):
        for agent_id, agent_path in agent_path_list.items():
            agent_path = list(agent_path.values())
            print(f"{agent_path}\n")
            agent = Agent(name=f"agent{agent_id}", path=agent_path, position=agent_path[:2], agent_info={"rx": 3, "ry": 1.5, "rad": agent_path[2]}, personalspace=10, size = 5, unique_num=int("{}".format(agent_id))
            )
            agent.isArrived = False
            agent.isStarted = True
            self.human_list.append(agent)
            self.human_name_index[f"{agent_id}"] = len(self.human_list) - 1
    
    def rvo2setup(self, agent_positions, neighbour_target=None,wall=None, steps=None, pattern=None, pattern_num=None):
        # あらかじめパスを指定する場合
        # agent_pass_list = makeHumanPath(len(agent_positions), steps + 1, wall, pos_list_args=agent_positions, radius=10, neighborDist=20)
        # agent_keys = list(agent_pass_list.keys())
        # for idx, agent_position in enumerate(agent_positions):
        #     yaw = 0
        #     if (len(agent_position) == 3):
        #         yaw = agent_position[2]
            
        #     agent = RVO2Human(name="smagv" + str(idx) + pattern, position=agent_position, personalspace=10, size = 5,agent_info={"rx":3, "ry": 1.5, "rad":yaw}, unique_num=int("{}{}".format(pattern_num, idx)))
        #     agent.setPath(agent_pass_list[agent_keys[idx]])
        #     self.human_list[agent_keys[idx]] = agent
        
        self.mode = 'RVO2'
        
        # 毎ステップシミュレートする場合
        self.sim = rvo2.PyRVOSimulator(10.0, 30.0, 6, 15.0, 20.0, 3.0, 10)
        
        facing_directions = {}
        positions_and_directions = []

        # エージェントの作成
        for idx, agent_position in enumerate(agent_positions):
            agent = None
            if idx < 8:
            # 最初の8人はほぼneighbour_targetを見ている状態
                agent = RVO2Human(
                    name=f"rvo2_human_{idx}",
                    sim=self.sim,
                    personalspace=10,
                    size=5,
                    agent_info={"rx": 3, "ry": 1.5, "rad": agent_position[2]},
                    unique_num=int(f"{idx}"),
                    position=agent_position,
                    neighbour_target=neighbour_target + np.array([random.uniform(-10, 10), random.uniform(-10, 10)]),
                    gaze_prob_offset=10,
                    satisfy_gazeing_a_offset=random.randint(300, 500)
                )
            elif 8 <= idx < 13:
                # 次の5人は途中からneighbour_targetを見るように設定
                agent = RVO2Human(
                    name=f"rvo2_human_{idx}",
                    sim=self.sim,
                    personalspace=10,
                    size=5,
                    agent_info={"rx": 3, "ry": 1.5, "rad": agent_position[2]},
                    unique_num=int(f"{idx}"),
                    position=agent_position,
                    neighbour_target=neighbour_target + np.array([random.uniform(-10, 10), random.uniform(-10, 10)]),
                    gaze_prob_offset=random.uniform(40, 90),
                    satisfy_gazeing_a_offset=random.randint(150, 200)
                )
            else:
                # 残りは何も見ずにdestinationに向かう、見ている人たちの後ろを素通りする経路を設定
                destination = np.array([agent_position[0] + random.uniform(-5, 5), agent_position[1] + 50])  # 後ろを素通りするような目的地設定
                agent = RVO2Human(
                    name=f"rvo2_human_{idx}",
                    sim=self.sim,
                    personalspace=10,
                    size=5,
                    agent_info={"rx": 3, "ry": 1.5, "rad": agent_position[2]},
                    unique_num=int(f"{idx}"),
                    position=agent_position,
                    destination=destination,
                    neighbour_target=None,
                    gaze_prob_offset=100
                )
            agent.isArrived = False
            agent.isStarted = True
            self.human_list.append(agent)
            self.human_name_index[f"{idx}"] = len(self.human_list) - 1
            facing_directions[agent.sim_agent] = agent_position[2]
        
    
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
    def __init__(self, name, sim, position, velocity="0", path=[],  personalspace=0,size = 10,color="k", agent_info = {}, unique_num=-1, dt = 0.1, destination=None, neighbour_target=None, gaze_prob_offset=70, satisfy_gazeing_a_offset=None, neighbour_target_prob=0.3, gaze_time_prob_range=(0, 0.5)):
        super().__init__(name, velocity, path, position, personalspace, size, color, agent_info, unique_num, dt)
        self.name = name
        self.sim = sim
        self.agent_id = unique_num
        self.position = np.array(position)
        self.state = -1  # 初期状態を0に設定
        self.step = 0
        self.gaze_time = 0
        self.active_step = random.randint(0, 50)
        self.gaze_prob_offset = gaze_prob_offset
        self.satisfy_gazeing_a_offset = random.randint(70, 80) if satisfy_gazeing_a_offset is None else satisfy_gazeing_a_offset
        self.neighbour_target_prob = neighbour_target_prob
        self.gaze_time_prob = random.uniform(*gaze_time_prob_range)
                
        self.is_satisfy_position = False
        self.is_statisfy_gazing_a = False
        self.is_already_gaze_target = False
        self.is_arrived = False
        
        self.sim_agent = sim.addAgent(tuple(self.position[:2]))
        
        self.destination = destination if destination is not None else np.array([random.uniform(-50, 130), random.uniform(-50, 130)])
        self.neighbour_target = neighbour_target 
        self.facing_direction = position[2]  # 初期の向き
        self.movement_direction = position[2]  # 初期の移動方向を向きと同じに設定

    def move(self):
        if self.state == 0:
            self.goDestination()
        elif self.state == 1:
            self.gazeTarget()
        elif self.state == 2:
            self.is_arrived = True
        elif self.state == 3:
            self.goNeighbourTarget()

        self.position = [*self.sim.getAgentPosition(self.sim_agent), self.movement_direction + np.pi / 2]

        self.changestate()
        self.step += 1
        

    def goDestination(self):
        # 目的地に向かって移動
        current_position = np.array(self.sim.getAgentPosition(self.sim_agent))
        direction = self.destination - current_position
        distance = np.linalg.norm(direction)
        if distance < 5.0:
            # 目的地に到着
            self.is_satisfy_position = True
            self.sim.setAgentPrefVelocity(self.sim_agent, (0.0, 0.0))
        else:
            velocity = direction / distance * 0.1 # 正規化
            self.sim.setAgentPrefVelocity(self.sim_agent, tuple(velocity))

    def goNeighbourTarget(self):
        # 路上パフォーマンスを行うターゲットに向かって移動
        if self.neighbour_target is None:
            current_position = np.array(self.sim.getAgentPosition(self.sim_agent))
            self.neighbour_target = current_position + np.array([random.uniform(-5, 5), random.uniform(-5, 5)])
        current_position = np.array(self.sim.getAgentPosition(self.sim_agent))
        direction = self.neighbour_target - current_position
        distance = np.linalg.norm(direction)

        # ターゲットの周囲15単位以内には近づかない
        min_distance_to_target = 20.0
        if distance < min_distance_to_target:
            # ターゲットに近づきすぎた場合は停止
            print(f"Too close to Neighbour Target: pose {current_position}, target {self.neighbour_target}, distance {distance}")
            self.is_satisfy_position = True
            self.sim.setAgentPrefVelocity(self.sim_agent, (0.0, 0.0))
            return
   
        # 距離に応じて確率を計算
        max_distance = 30.0  # 最大距離（調整可能）
        distance_ratio = min(distance / max_distance, 1.0)
        stop_probability = 1.0 - distance_ratio  # 距離が近いほど確率が高くなる

        if random.random() < stop_probability:
            # 確率的に停止
            print(f"Arrive Neighbour: pose {current_position}, target {self.neighbour_target}, distance {distance}")
            self.is_satisfy_position = True
            self.sim.setAgentPrefVelocity(self.sim_agent, (0.0, 0.0))
        else:
            velocity = direction / distance * 0.1 # 正規化
            self.sim.setAgentPrefVelocity(self.sim_agent, tuple(velocity))

    def gazeTarget(self):
        self.is_already_gaze_target = True
        # その場に留まる
        self.sim.setAgentPrefVelocity(self.sim_agent, (0.0, 0.0))
        
        self.gaze_time += 1
        temp_prob = random.random()
        adjusted_function = self.adjustedFunction(self.gaze_time, self.satisfy_gazeing_a_offset)
        if (temp_prob <= adjusted_function):
            print(f"Satisfy gazing a : prob {temp_prob}, adjusted_function {adjusted_function}, gaze_time {self.gaze_time}")
            self.gaze_time = 0
            self.is_statisfy_gazing_a = True
            self.gaze_prob_offset = 100

    @staticmethod
    def adjustedFunction(x, offset):
        # x が小さい場合は 0 に近く、大きくなると 1 に近づく関数
        y = 1 / (1 + np.exp(-0.5 * (x - offset)))
        return y

    def changestate(self):
        if self.state == -1:
            if (self.step >= self.active_step):
                self.state = 0
        elif self.state == 0:
            # 注視するかどうかを判断
            if (random.random() < self.adjustedFunction(self.step, self.gaze_prob_offset)) and (not self.is_already_gaze_target):
                self.state = 3  # 近陽のターゲットに移動
                self.is_satisfy_position = False
            if (self.is_satisfy_position):
                self.state = 2
        elif self.state == 3:
            # 位置が満たされたか確認
            if self.is_satisfy_position:
                self.state = 1  # 注視を開始
                self.gaze_time = 0
                self.is_statisfy_gazing_a = False
        elif self.state == 1:
            if not self.is_satisfy_position:
                self.state = 3  # 再度近陽のターゲットへ移動
            elif self.is_statisfy_gazing_a:
                self.state = 0  # 次の目的地へ
                self.is_satisfy_position = False

    def update_movement_direction(self):
        velocity = self.sim.getAgentVelocity(self.sim_agent)
        if np.linalg.norm(velocity) > 0:
            self.movement_direction = np.arctan2(velocity[1], velocity[0])
            self.facing_direction = self.movement_direction

# RVO2であらかじめ指定したパスを通る場合のクラス
class RVO2HumanOld(Agent):

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

