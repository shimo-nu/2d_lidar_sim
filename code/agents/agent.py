import os, sys

import numpy as np
from debug import debug_print

class Agent(object):
    isArrived = False
    isStarted = False
    # isStarted : 0, isArrived : 0 , stand-by
    # isStarted : 1, isArrived : 0, running
    # isStarted : 1, isArrived : 1 finish
    # velocity : (x軸方向の進む距離, y軸方向の進む距離)

    def __init__(self, name="test", velocity="0", path=[], position=[],  personalspace=0,size = 10,color="k", agent_info = {}, unique_num=-1, dt = 0.1,):
        self.velocity = velocity
        self.path = path
        self.init_path = np.array(path)
        self.position = np.array(position)
        self.init_position = np.array(position)
        self.personalspace = personalspace
        self.name = name
        self.size = size
        self.color = color
        self.unique_num = unique_num
        self.agent_info = agent_info
        self.dt = dt

    def isPersonalSpace(self, point):
        return False

    def setPosition(self, position):
        self.position = position

    def reset(self):
        self.position = list(self.init_position)
        self.path = self.init_path

    def checkInPersonalSpace(self, pose):
        if ((self.position[0] - pose[0])**2 + (self.position[1] - pose[1]**2) < self.personalspace**2):
            return True
        else:
            return False

    def taskmove(self):
        pass

    def move(self):
        #         if len(self.path) == 0:
        #             if (agent_pos[0] > self.position[0]):
        #                 self.position[0] += self.velocity[0]
        #             else:
        #                 self.position[0] -= self.velocity[0]
        #             self.position[1] = self.dcircle(self.position[0], agent_pos[0], agent_pos[1])
        #         else:
        if len(self.path) != 0:
            self.position = self.path[0]
            self.path = self.path[1:]

    def debug(self):
        print("==================Agent==================")
        debug_print("name is ", self.name)
        debug_print("path is ", self.path)
        debug_print("velocity is ", self.velocity)
        debug_print("position is ", self.position)
        debug_print("personalspace is ", self.personalspace)
        debug_print("unique_num is ", self.unique_num)
        print("=========================================")

