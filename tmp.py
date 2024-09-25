import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import one_azimuth_scan, makeWall, plot_test
from agents.human import HumanManager

def initialize_human_manager(agent_positions):
    """エージェントの初期化を行う"""
    human_manager = HumanManager()
    human_manager.nomove(agent_positions)
    return human_manager

def create_mapinfo():
    """地図の壁情報を生成する"""
    return [
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

def scan_and_update_map(mapinfo, new_mapinfo, robot_position, human_list, ogm, fig, ax):
    """センサーでスキャンし、地図情報を更新する"""
    show_axis_option = False
    ogm_info = {"size": [100, 100]}

    for destination in mapinfo:
        # plt.cla()
        agent_value_list = [[agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info]
                            for agent in human_list]

        scan_data = one_azimuth_scan(ogm_info, new_mapinfo, robot_position, destination, agent_value_list, 50)
        # plot(scan_data[0], fig, ax, mapinfo=new_mapinfo, agent_list=human_list, dst=destination, now=robot_position, axis_option=not show_axis_option)
        ogm *= scan_data[0]  # マップ更新
        show_axis_option = True
        # plt.pause(0.05)

def plot_final_map(ogm, new_mapinfo, human_list, robot_position, destination):
    """最終的なOGMとエージェント情報をプロットする"""
    fig, ax = plt.subplots()
    node_coordinates = plot_test(ogm, fig, ax, mapinfo=new_mapinfo, agent_list=human_list, dst=destination, now=robot_position, axis_option=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.show()

    # plot_2d_map(node_coordinates)

def plot_2d_map(node_coordinates, threshold_distance=10):
    G = nx.Graph()

    # ノードを追加
    for idx, coord in enumerate(node_coordinates):
        G.add_node(idx, pos=coord)

    for i in range(len(node_coordinates)):
        for j in range(i + 1, len(node_coordinates)):
            distance = np.linalg.norm(np.array(node_coordinates[i]) - np.array(node_coordinates[j]))
            
            if distance <= threshold_distance:
                G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')

    # グラフの描画
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, font_size=12, edge_color="gray")
    plt.show()

def main():
    # ロボットとエージェントの初期位置
    robot_position = [50, 50]
    agent_positions = [[70, 50, 0], [40, 50, 0], [80, 45, np.pi/4], [35, 20, np.pi/2], [65, 30, np.pi/3], [80, 30, np.pi/2]]

    # エージェントとマップ情報の初期化
    human_list = initialize_human_manager(agent_positions)
    mapinfo = makeWall(robot_position, 51)
    new_mapinfo = create_mapinfo()

    # OGM（Occupancy Grid Map）の初期化
    ogm = np.ones((100, 100))

    # センサーでスキャンし、マップ情報を更新
    fig, ax = plt.subplots()
    scan_and_update_map(mapinfo, new_mapinfo, robot_position, human_list, ogm, fig, ax)
    plt.close()

    # 最終的なOGMとエージェント情報をプロット
    print(ogm.shape)
    plot_final_map(ogm, new_mapinfo, human_list, robot_position, [40, 100])


if __name__ == "__main__":
    main()
