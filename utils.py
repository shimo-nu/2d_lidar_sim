import math
# import png
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import time
from gridmap import OccupancyGridMap
from matplotlib import patches
from matplotlib.ticker import MaxNLocator
from decimal import *
import math
from scipy.interpolate import interp1d, CubicSpline
import lidar_to_grid_map as lg
import joblib
from joblib import cpu_count


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__} : {elapsed_time} [min]")
        return result
    return wrapper
def makeWall(robot_position, lidar_radius):
        _wall = []
        left_upper = [robot_position[0] - lidar_radius,
                      robot_position[1] + lidar_radius]
        left_lower = [robot_position[0] - lidar_radius,
                      robot_position[1] - lidar_radius]
        right_upper = [robot_position[0] + lidar_radius,
                       robot_position[1] + lidar_radius]
        right_lower = [robot_position[0] + lidar_radius,
                       robot_position[1] - lidar_radius]
        _wall.extend([[left_lower[0], left_lower[1] + i]
                     for i in range(2*lidar_radius)])
        _wall.extend([[left_upper[0] + i, left_upper[1]]
                     for i in range(2*lidar_radius)])
        _wall.extend([[right_upper[0], right_upper[1] - i]
                     for i in range(2*lidar_radius)])
        _wall.extend([[right_lower[0] - i, right_lower[1]]
                     for i in range(2*lidar_radius)])
        return _wall

def dist2d(point1, point2):
    """
    Euclidean distance between two points
    :param point1:
    :param point2:
    :return:
    """

    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]

    dist2 = (x1 - x2)**2 + (y1 - y2)**2

    return math.sqrt(dist2)


def calcLinear(p1, p2):
    a = p2[1]-p1[1]
    b = p1[0]-p2[0]
    c = p1[1]*p2[0]-p1[0]*p2[1]
    return a, b, c


def calcEuclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def isPointInArea(point_list, area_center, area_dis):
    result = np.full(len(point_list), False)
    for point_idx in range(len(point_list)):
        if calcEuclidean(point_list[point_idx], area_center) < area_dis:
            result[point_idx] = True
    return result


def calcCircleLine(a, b, c, center, radius):
    D = abs(a * center[0] + b * center[1] + c)
    Ax = (a * D - b * math.sqrt((a*a + b*b) * radius *
          radius - D*D) / (a*a + b*b)) + center[0]
    Ay = (b * D + a * math.sqrt((a*a + b*b) * radius *
          radius - D*D) / (a*a + b*b)) + center[1]
    Bx = (a * D + b * math.sqrt((a*a + b*b) * radius *
          radius - D*D) / (a*a + b*b)) + center[0]
    By = (b * D - a * math.sqrt((a*a + b*b) * radius *
          radius - D*D) / (a*a + b*b)) + center[0]
    return [Ax, Ay], [Bx, By]


def calcDistance(a, b, c, point_x, point_y):  # 直線ax+by+c=0 点(x0,y0)
    numer = abs(a*point_x + b*point_y + c)  # 分子
    denom = math.sqrt(pow(a, 2)+pow(b, 2))  # 分母
    return numer/denom


def png_to_ogm(filename, normalized=False, origin='lower'):
    """
    Convert a png image to occupancy data.
    :param filename: the image filename
    :param normalized: whether the data should be normalised, i.e. to be in value range [0, 1]
    :param origin:
    :return:
    """
    r = png.Reader(filename)
    img = r.read()
    img_data = list(img[2])

    out_img = []
    bitdepth = img[3]['bitdepth']

    for i in range(len(img_data)):

        out_img_row = []

        for j in range(len(img_data[0])):
            if j % img[3]['planes'] == 0:
                if normalized:
                    out_img_row.append(img_data[i][j]*1.0/(2**bitdepth))
                else:
                    out_img_row.append(img_data[i][j])

        out_img.append(out_img_row)

    if origin == 'lower':
        out_img.reverse()

    return out_img


def plot_path(path):
    start_x, start_y = path[0]
    goal_x, goal_y = path[-1]

    # plot path
    path_arr = numpy.array(path)
    plt.plot(path_arr[:, 0], path_arr[:, 1], 'y')

    # plot start point
    plt.plot(start_x, start_y, 'ro')

    # plot goal point
    plt.plot(goal_x, goal_y, 'go')

    plt.show()

def plot(data, fig, ax, agent_list = None,dst = None, now = None,title = "", mapinfo = None,  grid_marks = None, filter_size = 1, axis_option=False):
    data_size = data.shape
    
    
    data_plot = ax.imshow(data, cmap="Purples",  vmin=0, vmax=1,origin='upper', extent=[0,data_size[0], data_size[1], 0],interpolation='none', alpha=1)
    if (agent_list is not None):
        for agent in agent_list:
            ax.plot(agent.position[0], agent.position[1],  c="#42F371", marker=".", markersize=20)
    
    # plt.grid(True, which="minor", color="w", linewidth=.6, alpha=0.5)
    if (grid_marks is not None):
        for grid_mark in grid_marks:
            r = patches.Rectangle([grid_mark[0] - int(filter_size / 2), grid_mark[1] - int(filter_size / 2)] , filter_size, filter_size, fill=True, edgecolor="blue", linewidth=3, label="rectangle")
            ax.add_patch(r)
    ax.set_title(title)
    if (dst is not None):
        ax.plot(dst[0], dst[1], 'x', c = "red",markersize=10)
    if (now is not None):
        ax.plot(now[0], now[1], '.', c = "blue", markersize=10)
    if (mapinfo is not None):
        x = [i[0] for i in mapinfo]
        y = [i[1] for i in mapinfo]
        ax.plot(x, y)

    if (axis_option):
        fig.colorbar(data_plot)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.invert_yaxis()

def plot_test(data, fig, ax, agent_list=None, dst=None, now=None, title="", mapinfo=None, grid_marks=None, filter_size=1, axis_option=False):
    data_size = data.shape
    node_coordinates = []  

    data_plot = ax.imshow(data, cmap="Purples", vmin=0, vmax=1, origin='upper', 
                          extent=[0, data_size[0], data_size[1], 0], interpolation='none', alpha=1)

    if agent_list is not None:
        for agent in agent_list:
            ax.plot(agent.position[0], agent.position[1], c="#42F371", marker=".", markersize=5)
    
    if grid_marks is not None:
        for grid_mark in grid_marks:
            r = patches.Rectangle([grid_mark[0] - int(filter_size / 2), grid_mark[1] - int(filter_size / 2)], 
                                  filter_size, filter_size, fill=True, edgecolor="blue", linewidth=3, label="rectangle")
            ax.add_patch(r)

    #ノードの可視化
    node_count = 10  
    node_size = data_size[0] // node_count  
    for i in range(node_count):
        for j in range(node_count):
            # 対応する ogm の値が 0 でない場合のみ点を描画
            ogm_x = int((i + 1/2) * node_size) 
            ogm_y = int((j + 1/2) * node_size)
            
            # ogm が 0 でない場合のみ描画
            if data[ogm_y, ogm_x] != 0:  
                ax.plot((i + 1/2) * node_size, (j + 1/2) * node_size, 'o', c="green", markersize=2)
                node_coordinates.append((ogm_x, ogm_y))

    ax.set_title(title)
    
    if dst is not None:
        ax.plot(dst[0], dst[1], 'x', c="red", markersize=10)
    
    if now is not None:
        ax.plot(now[0], now[1], '.', c="blue", markersize=10)
    
    if mapinfo is not None:
        x = [i[0] for i in mapinfo]
        y = [i[1] for i in mapinfo]
        ax.plot(x, y)
    
    if axis_option and fig is not None:
        fig.colorbar(data_plot)

    # 軸の設定
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.invert_yaxis()

    return node_coordinates

def doRough(data, stride = 2, filter_size = 2, isPlot=False) -> None:
    data_size = data.shape
    # new_ogm_size = [int(data.dim_cells[0] / slide_value), int(data.dim_cells[1] / slide_value)]
    new_ogm_size = [int((data_size[0] - filter_size) / stride) + 1, int((data_size[1] - filter_size) / stride) + 1]
    new_ogm = OccupancyGridMap(data_array = np.zeros(new_ogm_size), 
                        cell_size=1)
    x_idx, y_idx = 0, 0
    for x in range(new_ogm_size[0]):
        for y in range(new_ogm_size[1]):
            new_ogm.set_data((x, y), np.sum(data[stride * y : stride * y + filter_size, stride * x : stride * x + filter_size]) / (filter_size * filter_size))
    if (isPlot):
        plot(new_ogm.data, [])
    return new_ogm

def searchDst(data, stride, filter_size):
    data_size = data.shape
    if (data_size[0] == 1):
        return 0
    new_data = doRough(data, stride, filter_size)
    dst = searchDst(new_data, stride, filter_size)
    slice_data = []
    
    # console.log("###########################################")
    if (dst == 0):
        slice_data = data.data
        slice_data_argmax = np.unravel_index(np.argmax(slice_data), slice_data.shape)
        
        plot(data.data, [], title="data", grid_mark=slice_data_argmax, filter_size=1)
        return slice_data_argmax
    else:
        # console.log("slice index : [{}:{}, {}:{}]".format(stride * dst[0], stride * dst[0] + filter_size, stride * dst[1],stride * dst[1] + filter_size))
        slice_data = data.data[stride * dst[0] : stride * dst[0] + filter_size, stride * dst[1] : stride * dst[1] + filter_size]
  
    slice_data_argmax = np.unravel_index(np.argmax(slice_data), slice_data.shape)
    next_dst = (slice_data_argmax[0] + stride * dst[0], slice_data_argmax[1] + stride * dst[1])
    plot(data.data, [], title="data", grid_mark=next_dst, filter_size=2)
    plot(slice_data, [], title="slice_data", grid_mark=slice_data_argmax, filter_size=1)
    # console.log("###########################################")
    return next_dst

def searchDsts(data, stride, filter_size):
  x_size, y_size = data.shape
  if (filter_size > x_size or filter_size > y_size):
    raise ValueError("filter_size is larger than data size")
  poi_list = []
  for y_idx in range(int(y_size / filter_size)):
    for x_idx in range(int(x_size / filter_size)):
      poi_v = np.sum(data[y_idx * filter_size:(y_idx + 1) * filter_size-1,x_idx * filter_size:(x_idx+1) * filter_size-1])
      if (poi_v / (filter_size * filter_size) > 0.8):
        x = x_idx * filter_size + int(filter_size / 2)
        y = y_idx * filter_size + int(filter_size / 2)
        poi_list.append((x, y))
  return poi_list
def deleteDuplicatePoint(path):
    new_path = []
    before_position = []
    for pos in path:
        if (len(before_position) == 0):
            before_position = pos
            new_path.append(list(pos))
            continue
        # If the pos is at the same time as the previous measurement time
        if (before_position[0] == pos[0]):
            continue
        elif (before_position[0] + 1 == pos[0]):
            before_position = pos
            new_path.append(list(pos))
        else:
            # print(f'before_position : {before_position}')
            # print(f'now : {pos}')
            # print('jump time from previos measurement time')
            pass
    return new_path

def convertPathWithAffineMatrix(path, affine_matrix):
    convertFunc = lambda coordinate : [int(coordinate[0]), int(Decimal(str(coordinate[1]/1000)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)) + affine_matrix[0], int(Decimal(str(coordinate[2]/1000)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)) + affine_matrix[1]]
    new_path = list(map(convertFunc, path))
    return new_path

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def generate_pseudo_trajectory(grid_size, meters_per_grid=1.0, current_position = None, max_velocity=1.0, max_acceleration=0.1, wall_avoidance_prob=0.2, target_coords=None, random_seed = 123):

    np.random.seed(seed=random_seed)


    num_points = grid_size * grid_size

    # 初期位置をランダムに設定
    if (current_position is None):
        x0, y0 = np.random.uniform(30, 70), np.random.uniform(30, 70)
    else:
        x0, y0 = current_position


    # 初期速度と加速度をランダムに設定
    vx0, vy0 = np.random.uniform(-max_velocity, max_velocity), np.random.uniform(-max_velocity, max_velocity)
    ax, ay = np.random.uniform(-max_acceleration, max_acceleration), np.random.uniform(-max_acceleration, max_acceleration)

    # 位置・速度・加速度の配列を初期化
    x, y, vx, vy = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)

    # 初期値を代入
    x[0], y[0], vx[0], vy[0] = x0, y0, vx0, vy0

    # ランダムな擬似滞留データを生成
    for i in range(1, num_points):
        if target_coords is not None and np.random.random() < wall_avoidance_prob:
            # 指定された座標の近くに行くような動きを追加
            target_x, target_y = target_coords
            dx = target_x - x[i - 1]
            dy = target_y - y[i - 1]
            norm = np.linalg.norm([dx, dy])
            if norm > 0:
                vx[i] = max_velocity * dx / norm
                vy[i] = max_velocity * dy / norm
            else:
                vx[i] = vx[i - 1]
                vy[i] = vy[i - 1]
        else:
            vx[i] = vx[i - 1] + ax
            vy[i] = vy[i - 1] + ay

        # 位置を計算し、正の範囲内に制約
        x[i] = max(0, min(x[i - 1] + vx[i] * meters_per_grid, grid_size - 1))
        y[i] = max(0, min(y[i - 1] + vy[i] * meters_per_grid, grid_size - 1))

    # グリッドに変換して返す
    x_grid = np.floor(x / meters_per_grid).astype(int)
    y_grid = np.floor(y / meters_per_grid).astype(int)


    # 移動方向の角度を計算
    directions = np.arctan2(vx, vy)

    return x_grid, y_grid, directions

def createMask(coordinate, value, location):
    isInside = point_in_polygon(coordinate, location)
    if (isInside):
        return value
    else:
        return 0

# import rvo2
import random

def dst2vel(pos, dst, offset = 10):
    norm = np.linalg.norm(pos - dst)
    return tuple((dst - pos) * offset / norm)

def isReachedGoal(pos, dst, radius):
    return np.linalg.norm(np.array(dst, dtype=np.float16) - np.array(list(pos), dtype=np.float16)) < radius

def clampOnRange(x, min_v, max_v):
    if (x < min_v):
        return min_v
    elif (x > max_v):
        return max_v
    else:
        return x
def clampOnRectangle(point, rectangle):
    clamp = [0,0]
    clamp[0] = clampOnRange(point[0], rectangle['origin'][0], rectangle['origin'][0] + rectangle['size'][0])
    clamp[1] = clampOnRange(point[1], rectangle['origin'][1], rectangle['origin'][0] + rectangle['size'][1])
    return clamp
def circleRectangleCollision(circle, rectangle):
    clamped = clampOnRectangle(circle['center'], rectangle)
    if (np.linalg.norm(np.array(circle['center']) - np.array(clamped)) < circle['radius']):
        return True
    return False

def makeHumanPath(num_agents, steps, obstacles, dst_list_args = None, pos_list_args = None, time_width = 10,  timeStep = 1, neighborDist = 5, maxNeighbors = 5, timeHorizon = 1.5, timeHorizonObst = 2, radius = 0.4, maxSpeed = 2):

    dst_list = []
    pos_list = []

    for num in range(num_agents):
        if (dst_list_args is None):
            dst_list.append(np.array([random.randrange(10, 30), random.randrange(10, 30)]))
        else:
            dst_list = dst_list_args
        if (pos_list_args is None):
            pos_list.append(np.array([random.randrange(10, 30), random.randrange(10, 30)]))
        else:
            # pos_list_args has three dimensions, position.xy and yaw, it convert dimensions from three(position, yaw) to two(only position).
            pos_list = [position[:2] for position in pos_list_args]
    sim = rvo2.PyRVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed)

    agents_list = []
    agents_dst_list = {}
    for num in range(num_agents):
        agents_list.append(sim.addAgent(tuple(pos_list[num])))

    for idx, agent in enumerate(agents_list):
        sim.setAgentPrefVelocity(agent, dst2vel(pos_list[idx], dst_list[idx]))
        agents_dst_list[agent] = dst_list[idx]


    o1 = sim.addObstacle(obstacles)
    sim.processObstacles()


    pass_list = {agent_no:[sim.getAgentPosition(agent_no)] for agent_no in agents_list}
    goal_check_radius = 10
    is_agent_new_dst = {agent_no:float('inf') for agent_no in agents_list}
    agent_re_route_cnt = {agent_no:0 for agent_no in agents_list}
    for step in range(steps):
        sim.doStep()

        for agent_no in agents_list:
            position = sim.getAgentPosition(agent_no)[:]
            if (math.isnan(position[0])):
                pass_list[agent_no].append(pass_list[agent_no][-1][:])
            else:
                pass_list[agent_no].append(position)
            if (agent_no > 3):
                continue

            if (is_agent_new_dst[agent_no] == 0):
                if (agent_re_route_cnt[agent_no] == 0):
                    new_dst = np.array([random.randrange(60, 99), random.randrange(60, 99)])
                elif (agent_re_route_cnt[agent_no] == 1):
                    new_dst = np.array([random.randrange(0, 99), random.randrange(0, 99)])
                sim.setAgentPrefVelocity(agent_no, dst2vel(pass_list[agent_no][-1], new_dst))
                # print("change destination of {} from {} to {}".format(agent_no, agents_dst_list[agent_no], new_dst))
                agents_dst_list[agent_no] = new_dst[:]
                is_agent_new_dst[agent_no] = float('inf')
            if (is_agent_new_dst[agent_no] == float('inf') and isReachedGoal(pass_list[agent_no][-1], agents_dst_list[agent_no], goal_check_radius)):
                sim.setAgentPrefVelocity(agent_no, (0, 0))
                is_agent_new_dst[agent_no] = time_width
                print(agent_no)



        positions = [sim.getAgentPosition(agent_no)
                 for agent_no in agents_list]
        # [(x, 'odd') if x % 2 else (x, 'even') for x in r]
        positions_checkgoal = ["Goal" if isReachedGoal(sim.getAgentPosition(agent_no), agents_dst_list[agent_no], goal_check_radius) else sim.getAgentPosition(agent_no) for agent_no in agents_list]
        # print('step={}  t={}  {}'.format(step, sim.getGlobalTime(), sim.getAgentPosition(agent_no)))
    print(is_agent_new_dst)

    return pass_list


def is_inside_ellipse(point, ellipse):
    px = point[0] - ellipse['center'][0]
    py = point[1] - ellipse['center'][1]
    
    # angle = math.radians(ellipse['rad'])
    angle = ellipse['rad']
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    transformed_x = (px * cos_angle + py * sin_angle) / ellipse['rx']
    transformed_y = (px * sin_angle - py * cos_angle) / ellipse['ry']
    
    distance = transformed_x ** 2 + transformed_y ** 2
    
    return distance <= 1

def is_point_inside_rectangle(point_x, point_y, rect_top_left, rect_bottom_right):
    return rect_top_left[0] <= point_x <= rect_bottom_right[0] and rect_top_left[1] <= point_y <= rect_bottom_right[1]


from decimal import Decimal, ROUND_HALF_UP

def decimal_round(number, precision=0):
    """
    正確な四捨五入を行う関数
    :param number: 四捨五入する数値
    :param precision: 小数点以下の桁数（デフォルトは0）
    :return: 四捨五入した数値
    """
    decimal_number = Decimal(str(number))
    rounded_number = decimal_number.quantize(Decimal('0.' + '0' * precision), rounding=ROUND_HALF_UP)
    return float(rounded_number)


def generate_grid_points_in_ellipse(a, b, center_x, center_y, angle_rad, precision=1):
    # 楕円内の格子点を生成する関数

    grid_points = []
    step = 10 ** -precision

    # 楕円内にあるか確認して格子点をリストに追加
    for x in np.arange(center_x - a, center_x + a + step, step):
        for y in np.arange(center_y - b, center_y + b + step, step):
            x_rot = (x - center_x) * math.cos(angle_rad) + (y - center_y) * math.sin(angle_rad)
            y_rot = -(x - center_x) * math.sin(angle_rad) + (y - center_y) * math.cos(angle_rad)

            # 境界上にある格子点も追加
            grid_point = [round(x), round(y)]
            if abs((x_rot / a) ** 2 + (y_rot / b) ** 2 - 1) < 1e-5: # 誤差許容値を調整
                if (grid_point not in grid_points):
                    grid_points.append(grid_point)
            elif (x_rot / a) ** 2 + (y_rot / b) ** 2 < 1:
                if (grid_point not in grid_points):
                    grid_points.append(grid_point)

    return grid_points


def calculate_smooth_angle(trajectory):
    x_coords, y_coords = zip(*trajectory)
    t = np.arange(len(trajectory))
    f_x = CubicSpline(t, x_coords)
    f_y = CubicSpline(t, y_coords)

    angles = []
    for i in range(1, len(trajectory) - 1):
        x_prev, y_prev = f_x(i - 1), f_y(i - 1)
        x_curr, y_curr = f_x(i), f_y(i)
        x_next, y_next = f_x(i + 1), f_y(i + 1)

        angle_prev = math.atan2(y_curr - y_prev, x_curr - x_prev)
        angle_next = math.atan2(y_next - y_curr, x_next - x_curr)

        # Calculate the angle with respect to the positive x-axis
        angle = (angle_next - angle_prev) % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        angles.append(angle)
    angles.append(angle)
    return angles

def calculate_angle(trajectory):
    angles = []
    for i in range(1, len(trajectory) - 1):
        x_prev, y_prev = trajectory[i - 1]
        x_curr, y_curr = trajectory[i]
        x_next, y_next = trajectory[i + 1]

        angle_prev = math.atan2(y_curr - y_prev, x_curr - x_prev)
        angle_next = math.atan2(y_next - y_curr, x_next - x_curr)

        # Calculate the angle with respect to the positive x-axis
        angle = (angle_next - angle_prev) % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        angles.append(angle)
    angles.append(angle)
    
    return angles



def sample_rotated_ellipse_points(center, rx, ry, angle, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # 回転行列を計算
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # 回転前の楕円のx軸およびy軸上の座標を計算
    x_points = rx * cos_angles
    y_points = ry * sin_angles

    # 回転行列を使って座標を回転させる
    rotated_points = np.dot(rotation_matrix, np.vstack((x_points, y_points)))

    # 中心座標を加算して回転後の楕円の座標を計算
    x_rotated_points = center[0] + rotated_points[0, :]
    y_rotated_points = center[1] + rotated_points[1, :]

    return list(zip(x_rotated_points, y_rotated_points))


def one_azimuth_scan(ogm_info, mapinfo, robot_position, target_point, agent_list, lidar_radius):
    ogm_data = np.ones((ogm_info["size"][0], ogm_info["size"][1]))
    ogm = OccupancyGridMap(ogm_data, 1)

    # Find intersection point with walls
    occulusion_agent_list = []
    cross_map_wall_point = target_point

    for idx_mapinfo in range(1, len(mapinfo)):
        wall_start = mapinfo[idx_mapinfo - 1]
        wall_end = mapinfo[idx_mapinfo]
        is_intersect = isIntersect(
            robot_position, target_point, wall_start, wall_end)
        if (is_intersect and target_point[0] > 0 and target_point[1] > 0):
            try:
                _cross_map_wall_point = line_cross_point(
                    robot_position, target_point, wall_start, wall_end)
                if (_cross_map_wall_point is None):
                    continue
                elif (cross_map_wall_point == target_point or calcEuclidean(robot_position, _cross_map_wall_point) <= calcEuclidean(robot_position, cross_map_wall_point)):
                    cross_map_wall_point = _cross_map_wall_point
            except:
                # print("robot_pos : {}".format(robot_position))
                # print("target_pos : {}".format(target_point))
                # print("first_point : {}".format(first_point))
                # print("second_point : {}".format(second_point))
                # print("mapinfo : {}".format(mapinfo))
                pass


    # Find occluding agent

    # try:
    robot_position_ru = [int(decimal_round(robot_position[0])), int(decimal_round(robot_position[1]))]

    cross_map_wall_point_ru = [int(decimal_round(cross_map_wall_point[0]))- 1, int(decimal_round(cross_map_wall_point[1])) - 1]
    try:
        laser_beams = lg.bresenham(
            robot_position_ru, cross_map_wall_point_ru)
    except Exception as e:
        print("cross_point : {}, robot_position : {}, cmwallpt : {}, target : {}".format(cross_map_wall_point, robot_position, cross_map_wall_point, target_point))
        print(type(robot_position_ru[0]))
        print(type(robot_position_ru[1]))
        print(type(cross_map_wall_point_ru[0]))
        print(type(cross_map_wall_point_ru[0]))
        print(str(e))
        import sys
        sys.exit(1)
    occulusion_agent = [10000, None]
    isOcculusion_laser_beams = [False for i in range(len(laser_beams))]
    for agent in agent_list:
        # if (not agent.isStarted or agent.isArrived):
        if (not agent[1] or agent[0]):
            continue
            
        # checkOcculusion = [calcEuclidean(agent[2], laser_beam) < (
            # agent[3]) for laser_beam in laser_beams]
        ellipse = {'center' : agent[2], 'rx' : agent[5]['rx'], 'ry' : agent[5]['ry'], 'rad' : agent[5]['rad']}
        checkOcculusion = [is_inside_ellipse(laser_beam, ellipse) for laser_beam in laser_beams]
        if any(checkOcculusion):
            isOcculusion_laser_beams = [x | y for x, y in zip(
                isOcculusion_laser_beams, checkOcculusion)]
            occ_idx = checkOcculusion.index(True)
            if (occ_idx < occulusion_agent[0] and calcEuclidean(agent[2], robot_position) < lidar_radius):
                occulusion_agent[0] = occ_idx
                occulusion_agent[1] = agent[4]

    laser_point_list = laser_beams
    if any(isOcculusion_laser_beams):
        lb_idx = isOcculusion_laser_beams.index(True)
        laser_point_list = laser_beams[:lb_idx]

        if occulusion_agent[1] is not None and occulusion_agent[1] not in occulusion_agent_list:
            occulusion_agent_list.append(occulusion_agent[1])

    for laser_beam in laser_point_list:
        if (calcEuclidean(laser_beam, robot_position) > lidar_radius or np.all(laser_beam == robot_position)):
            continue
        # if (laser_beam.tolist() in self.ogm.mapinfo):
        #     break
        try:
            ogm.set_data(laser_beam, 0)
        except:
            pass


    # tm.result()
    return ogm.data, occulusion_agent_list# , tm.getTime()

def lidar_scan(lidar_position, area_size, mapinfo, lidar_radius, agent_list):
    ogm_data = np.ones((area_size[0], area_size[1]))
    ogm = OccupancyGridMap(ogm_data, 1)
    _wall = makeWall(lidar_position, lidar_radius)
    human_unique_num_dict = {
        agent.unique_num: agent for agent in agent_list}

    ogm_info = {"size": area_size}


    agent_value_list = [[agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info] for agent in agent_list]
    results = joblib.Parallel(n_jobs=get_availble_cpus(0.8), verbose=0, backend='threading')(joblib.delayed(one_azimuth_scan)(
        ogm_info, mapinfo, lidar_position, point, agent_value_list, lidar_radius) for point in _wall)


    # time = np.array([0., 0., 0.])
    for result in results:
        ogm.data *= result[0]
        # for _occulusion_agent in result[1]:
        #     if (_occulusion_agent not in self.count_people):
        #         count_people[_occulusion_agent] = human_unique_num_dict[_occulusion_agent]
        # time += result[2]
    
    return ogm_data

def isIntersect(a, b, c, d):
            tc = (a[0] - b[0]) * (c[1] - a[1]) - (a[1] - b[1]) * (c[0] - a[0])
            td = (a[0] - b[0]) * (d[1] - a[1]) - (a[1] - b[1]) * (d[0] - a[0])
            if (tc * td > 0):
                return False

            tc = (c[0] - d[0]) * (a[1] - c[1]) - (c[1] - d[1]) * (a[0] - c[0])
            td = (c[0] - d[0]) * (b[1] - c[1]) - (c[1] - d[1]) * (b[0] - c[0])
            if (tc * td > 0):
                return False
            return True

def line_cross_point(P0, P1, Q0, Q1):
    x0, y0 = P0
    x1, y1 = P1
    x2, y2 = Q0
    x3, y3 = Q1
    a0 = x1 - x0
    b0 = y1 - y0
    a2 = x3 - x2
    b2 = y3 - y2

    d = a0*b2 - a2*b0
    if d == 0:
        # two lines are parallel
        return None

    # s = sn/d
    sn = b2 * (x2-x0) - a2 * (y2-y0)
    # t = tn/d
    # tn = b0 * (x2-x0) - a0 * (y2-y0)
    return int(Decimal(str(x0 + a0*sn/d)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)), int(Decimal(str(y0 + b0*sn/d)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))



def project_point(a, b, d):
    # ベクトルABとADを計算
    ab = np.array(b) - np.array(a)
    ad = np.array(d) - np.array(a)
    
    # ベクトルABの長さの2乗
    ab_squared_norm = np.dot(ab, ab)
    
    # 点Dの直線ABへの射影点Pを計算
    projection_scalar = np.dot(ad, ab) / ab_squared_norm
    p = np.array(a) + projection_scalar * ab
    
    return p

def is_projection_closer_to_a(a, b, c, d):
    # 射影点Pを計算
    p = project_point(a, b, d)
    # ベクトルACとAPを計算
    ac = np.array(c) - np.array(a)
    ap = p - np.array(a)
    
    ac_norm = np.linalg.norm(ac)
    ap_norm = np.linalg.norm(ap)
    
    if (ac_norm < ap_norm):
        return False
    else:
        return True

def is_robot_aligned_with_line(a, b, robot_direction):
    # ベクトルABを計算
    vector_ab = np.array(b) - np.array(a)
    
    # ベクトルABとロボットの向きベクトルの正規化
    vector_ab_normalized = vector_ab / np.linalg.norm(vector_ab)
    robot_direction_normalized = robot_direction / np.linalg.norm(robot_direction)
    
    # 2つのベクトル間の内積を計算
    dot_product = np.dot(vector_ab_normalized, robot_direction_normalized)
    
    # 内積が1に近いかどうかをチェック（小さな誤差を許容）
    return dot_product > 0.999

def get_availble_cpus(cpu_ratio):
    available_cores = cpu_count()
    num_threads = max(1, int(available_cores * cpu_ratio))
    return num_threads


def is_agent_between_points(point1, point2, agent_list, col_meter):
    x1, y1 = point1
    x2, y2 = point2

    # 2点間の距離
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # 2点間のベクトル
    dx = (x2 - x1) / distance
    dy = (y2 - y1) / distance

    for agent in agent_list:
        # エージェントと点1の距離
        d1 = ((agent.position[0] - x1) ** 2 + (agent.position[1] - y1) ** 2) ** 0.5
        # エージェントと点2の距離
        d2 = ((agent.position[0] - x2) ** 2 + (agent.position[1] - y2) ** 2) ** 0.5

        # エージェントが2点の外側にある場合はスキップ
        if d1 > distance + col_meter or d2 > distance + col_meter:
            continue

        # エージェントと直線の最短距離
        perp_distance = abs(dy * agent.position[0] - dx * agent.position[1] + x2 * y1 - y2 * x1) / distance

        # 最短距離がcollision範囲内であれば、エージェントが2点間にいると判定
        if perp_distance <= col_meter:
            return True

    return False

from scipy.special import ellipe
class Ellipse:
    # 円周率の近似値をクラス変数として定義
    pi = np.pi

    # インスタンス変数の定義
    def __init__(self, a, b, center=(0,0), n=129):
        self.major_radius = a  # 長軸半径
        self.minor_radius = b  # 短軸半径
        self.center = center  # 中心点の座標
        self.f_distance = np.sqrt(a**2 - b**2)  # 原点から焦点までの距離
        
        # 右側焦点の座標と左側焦点の座標
        self.focus_1 = (self.center[0] + self.f_distance, self.center[1])
        self.focus_2 = (self.focus_1[0] - 2 * self.f_distance, self.center[1])
        
        self.eccentricity = self.f_distance / a  # 離心率
        self.area =self.pi * self.major_radius * self.minor_radius  # 面積
        self.perimeter = 4 * ellipe(self.eccentricity)  # 周の長さ
        
        t = np.linspace(0, 2*np.pi, n)
        self.x = self.center[0] + self.major_radius * np.cos(t)
        self.y = self.center[1] + self.minor_radius * np.sin(t)

    # 回転メソッド
    # angleで回転角度(ラジアン)を指定
    # r_axisで回転軸の座標を指定
    def rotate(self, angle, r_axis):

        # 回転行列
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])

        # 焦点の回転
        p = np.array(r_axis)
        self.focus_1 = np.dot(R, self.focus_1 - p) + p
        self.focus_2 = np.dot(R, self.focus_2 - p) + p

        # 楕円の回転
        q = np.array([[r_axis[0]],[r_axis[1]]])
        arr = np.vstack((self.x, self.y))
        arr_2 = np.dot(R, arr - q) + q
        self.x = arr_2[0]
        self.y = arr_2[1]

    # データ一覧表示メソッド
    def show_data(self):
        print(" 長軸半径　{}\n".format(self.major_radius),
              "短軸半径　{}\n".format(self.minor_radius),
              "中心点　{}\n".format(self.center),
              "第1焦点　{}\n".format(self.focus_1),
              "第2焦点　{}\n".format(self.focus_2),
              "離心率　{}\n".format(self.eccentricity),
              "面積　{}\n".format(self.area),
              "弧長　{}\n".format(self.perimeter))

    # 楕円の描画メソッド
    # oval_colorで楕円の線の色を指定
    # display_focusをTrueに設定すると焦点を表示
    # display_centerをTrueに設定すると中心点を表示
    def draw(self, ax, n=129, linestyle="-", oval_color="black",
             display_focus=False, focus_color="black",
             display_center=False, center_color="black",
             fill=False, fill_color="blue"):

        ax.plot(self.x, self.y, color=oval_color, linestyle=linestyle)
        # 楕円内の塗りつぶし
        if fill:
            # Yの上限と下限を計算
            y_upper = self.center[1] + self.minor_radius * np.sin(np.linspace(0, np.pi, n))
            y_lower = self.center[1] - self.minor_radius * np.sin(np.linspace(0, np.pi, n))
            # 楕円内部を塗りつぶし
            ax.fill_between(self.x, y_lower, y_upper, color=fill_color, alpha=0.3)

        if display_focus == True:
            ax.scatter(self.focus_1[0], self.focus_1[1], color=focus_color)
            ax.scatter(self.focus_2[0], self.focus_2[1], color=focus_color)
        
        if display_center == True:
            ax.scatter(self.center[0], self.center[1], color=center_color)
class TimeMeasure:
    def __init__(self, max_cp):
        self.cp_total_time = [0 for _ in range(max_cp)]
        self.cp_count = [0 for _ in range(max_cp)]
        self.cp_name = [i + 1 for i in range(max_cp)]
        self.last_measure_time = [None for i in range(max_cp)]
        self.max_cp = max_cp
        self.start_time = None
        self.lap_time = None
        self.title = ""

    def __enter__(self, title):
        self.title = title
        self.ftime = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        print("------- TimeMeasure : {} -------".format(self.title))
        print("Time : {}[s]".format(time.perf_counter() - self.ftime))

    def start(self):
        now_time = time.perf_counter()
        self.start_time, self.lap_time = now_time, now_time

    def setCPName(self, cp, name):
        self.cp_name[cp - 1] = name

    def setTitle(self, title):
        self.title = title
        
    def from_cp(self, cp_num):
        if (self.last_measure_time[cp_num - 1] is None):
            self.last_measure_time[cp_num - 1] = time.perf_counter()
            self.cp_count[cp_num - 1] += 1
            return
        
    def to_cp(self, cp_num):
        if (self.last_measure_time[cp_num - 1] is None):
            return

        self.cp_total_time[cp_num - 1] += (time.perf_counter() - self.last_measure_time[cp_num - 1])
        self.last_measure_time[cp_num - 1] = None
        
    def checkpoint(self, cp_num, title=""):
        if (self.start_time is None):
            self.start_time = time.perf_counter()
            print("Time did not start. So start.")
            return 
        print(f"Title ; {title}")
        print(f"Elapsed Time is {round(time.perf_counter() - self.start_time, 3)}")
        print(f"Lap Time is {round(time.perf_counter() - self.lap_time, 3)}")
        self.lap_time = time.perf_counter()

    def result(self):
        if (self.title != ""):
            print("------- TimeMeasure : {} -------".format(self.title))
        for idx, cp_time in enumerate(self.cp_total_time):
            if (self.cp_count[idx] == 0):
                print("[checkpoint] {} is not executed".format(idx+1))
            else:
                print("[checkpoint] {} : exec time, {}[s] exec count, {}".format(
                    self.cp_name[idx], round(cp_time / self.cp_count[idx], 4), self.cp_count[idx]))

    def getTime(self):
        return np.array([cp_time/self.cp_count[idx] if self.cp_count[idx] != 0 else 0 for idx, cp_time in enumerate(self.cp_total_time)], dtype=np.float32)



