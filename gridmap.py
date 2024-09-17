import numpy as np
import math
import matplotlib.pyplot as plt
# from lib.utils import *
import lidar_to_grid_map as lg
# from agent_define import Agent


class OccupancyGridMap:
    def __init__(self, data_array, cell_size, mapinfo=None, occupancy_threshold=0.8):
        """
        Creates a grid map
        :param data_array: a 2D array with a value of occupancy per cell (values from 0 - 1)
        :param cell_size: cell size in meters
        :param occupancy_threshold: A threshold to determine whether a cell is occupied or free.
        A cell is considered occupied if its value >= occupancy_threshold, free otherwise.
        """

        self.data = data_array
        self.mapinfo = mapinfo
        self.init_data = np.copy(data_array)
        self.dim_cells = data_array.shape
        self.dim_meters = (
            self.dim_cells[0] * cell_size, self.dim_cells[1] * cell_size)
        self.cell_size = cell_size
        self.occupancy_threshold = occupancy_threshold
        # 2D array to mark visited nodes (in the beginning, no node has been visited)
        self.visited = np.zeros(self.dim_cells, dtype=np.float32)

    def mark_visited_idx(self, point_idx):
        """
        Mark a point as visited.
        :param point_idx: a point (x, y) in data array
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        self.visited[y_index][x_index] = 1.0

    def mark_visited(self, point):
        """
        Mark a point as visited.
        :param point: a 2D point (x, y) in meters
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.mark_visited_idx((x_index, y_index))

    def is_visited_idx(self, point_idx):
        """
        Check whether the given point is visited.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is visited, false otherwise
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        if self.visited[y_index][x_index] == 1.0:
            return True
        else:
            return False

    def is_visited(self, point):
        """
        Check whether the given point is visited.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is visited, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_visited_idx((x_index, y_index))

    def get_data_idx(self, point_idx):
        """
        Get the occupancy value of the given point.
        :param point_idx: a point (x, y) in data array
        :return: the occupancy value of the given point
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        return self.data[y_index][x_index]

    def get_data(self, point):
        """
        Get the occupancy value of the given point.
        :param point: a 2D point (x, y) in meters
        :return: the occupancy value of the given point
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.get_data_idx((x_index, y_index))

    def set_data_idx(self, point_idx, new_value):
        """
        Set the occupancy value of the given point.
        :param point_idx: a point (x, y) in data array
        :param new_value: the new occupancy values
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            # print(x_index)
            # print(y_index)
            raise Exception('Point is outside map boundary')

        self.data[y_index][x_index] = new_value

    def set_data(self, point, new_value):
        """
        Set the occupancy value of the given point.
        :param point: a 2D point (x, y) in meters
        :param new_value: the new occupancy value
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        self.set_data_idx((x_index, y_index), new_value)

    def is_inside_idx(self, point_idx):
        """
        Check whether the given point is inside the map.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is inside the map, false otherwise
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            return False
        else:
            return True

    def is_inside(self, point):
        """
        Check whether the given point is inside the map.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is inside the map, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_inside_idx((x_index, y_index))

    def is_occupied_idx(self, point_idx):
        """
        Check whether the given point is occupied according the the occupancy threshold.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is occupied, false otherwise
        """
        x_index, y_index = point_idx
        if self.get_data_idx((x_index, y_index)) >= self.occupancy_threshold:
            return True
        else:
            return False

    def is_occupied(self, point):
        """
        Check whether the given point is occupied according the the occupancy threshold.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is occupied, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_occupied_idx((x_index, y_index))

    def get_index_from_coordinates(self, x, y):
        """
        Get the array indices of the given point.
        :param x: the point's x-coordinate in meters
        :param y: the point's y-coordinate in meters
        :return: the corresponding array indices as a (x, y) tuple
        """
        x_index = int(round(x/self.cell_size))
        y_index = int(round(y/self.cell_size))

        return x_index, y_index

    def get_coordinates_from_index(self, x_index, y_index):
        """
        Get the coordinates of the given array point in meters.
        :param x_index: the point's x index
        :param y_index: the point's y index
        :return: the corresponding point in meters as a (x, y) tuple
        """
        x = x_index*self.cell_size
        y = y_index*self.cell_size

        return x, y

    def get_size(self):
        return self.dim_cells

    def reset_data(self):
        self.data = np.copy(self.init_data)

    def setInitData(self, data=None):
        if data is None:
            self.init_data = np.copy(self.data)
        else:
            self.init_data = np.copy(data)

    def calculateBlindSpot(self, robot_pos, agent_list, threshold):
        self.reset_data()
        _occulusion_agent_list = []
        size = self.get_size()

        for x in range(size[0]):
            for y in range(size[1]):
                if ((x == robot_pos[0]) and (y == robot_pos[1])):
                    continue
                isOcculusion = False
                min_agent_target_dis = float("inf")
                near_agent_dict = dict()
                near_agent = Agent()
                for _agent in agent_list:
                    sign_agent = (np.array(_agent.position) -
                                  np.array(robot_pos))
                    sign_cell = (np.array([x, y]) - np.array(robot_pos))
                    sign = sign_agent * sign_cell

                    if (sign[0] < 0 or sign[1] < 0):
                        continue
                    a, b, c = calcLinear([x, y], robot_pos)
                    distance = calcDistance(
                        a, b, c, _agent.position[0], _agent.position[1])
                    if (_agent.personalspace + threshold > distance):
                        isOcculusion = True
                        if (_agent.name not in _occulusion_agent_list):
                            _occulusion_agent_list.append(_agent)
                        agent_target_dis = calcEuclidean(
                            robot_pos, _agent.position)
                        # near_agent_dict[_agent] = agent_target_dis
                        if (min_agent_target_dis > agent_target_dis):
                            min_agent_target_dis = agent_target_dis
                            # near_agent_dict[_agent] =  agent_target_dis

                            near_agent = _agent
                        # near_agent = _agent
                        # break
                p1 = []
                p2 = []

                if (isOcculusion):
                    if (calcEuclidean(robot_pos, [x, y]) <= calcEuclidean(robot_pos, near_agent.position)):
                        p1 = list(map(round, robot_pos))
                        p2 = [int(x), int(y)]
                    else:
                        p1 = list(map(round, robot_pos))
                        p2 = list(map(round, near_agent.position))
                else:
                    p1 = list(map(round, robot_pos))
                    p2 = [int(x), int(y)]
                laser_beams = lg.bresenham((p1[0], p1[1]), (p2[0], p2[1]))
                for laser_beam in laser_beams:
                    self.set_data(laser_beam, 0)
        return self.data

    def plot(self, robot_pos=[0, 0], agent_list=[], alpha=1, min_val=0, origin='lower'):
        """
        plot the grid map
        """

        plt.figure(figsize=(20, 8))
        plt.subplot(122)
        # Plot DATA
        plt.imshow(self.data, cmap="Purples", vmin=min_val, vmax=1,
                   origin=origin, interpolation='none', alpha=alpha)

        # Plot Agent
        for _agent in agent_list:
            plt.scatter(
                _agent.position[0], _agent.position[1], c=_agent.color, s=100)

        # Plot Robot
        plt.scatter(robot_pos[0], robot_pos[1], c="y",
                    edgecolor="orange", marker=",", s=300)

        plt.grid(True, which="minor", color="w", linewidth=.6, alpha=0.5)
        plt.clim(0, 1)
        plt.colorbar()
        # plt.draw()
        plt.show()

    # def draw_movie

    @staticmethod
    def from_png(filename, cell_size):
        """
        Create an OccupancyGridMap from a png image
        :param filename: the image filename
        :param cell_size: the image pixel size in meters
        :return: the created OccupancyGridMap
        """
        ogm_data = png_to_ogm(filename, normalized=True)
        ogm_data_arr = np.array(ogm_data)
        ogm = OccupancyGridMap(ogm_data_arr, cell_size)

        return ogm

