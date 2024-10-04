import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import one_azimuth_scan, makeWall, plot_test, extract_nodes_id
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
        [100, 100],
        [0, 100],
        [0, 0]
    ]

def scan_and_update_map(mapinfo, new_mapinfo, robot_position, human_list, ogm):
    """センサーでスキャンし、地図情報を更新する"""
    ogm_info = {"size": [100, 100]}
    
    for destination in mapinfo:
        agent_value_list = [[agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info]
                            for agent in human_list]
        scan_data = one_azimuth_scan(ogm_info, new_mapinfo, robot_position, destination, agent_value_list, 50)
        ogm *= scan_data[0]  # マップをスキャンデータで更新

def plot_final_map(ogm, new_mapinfo, human_list, robot_position, destination, ax, vis_node):
    """最終的なOGMとエージェント情報をプロットする"""
    plot_test(ogm, None, ax, vis_node, mapinfo=new_mapinfo, agent_list=human_list, dst=destination, now=robot_position, axis_option=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

def main():
    # ロボットとエージェントの初期位置
    agent_positions = [[70, 50, 0], [40, 50, 0], [80, 45, np.pi/4], [35, 20, np.pi/2], [65, 30, np.pi/3], [80, 30, np.pi/2]]

    node_count = 10  # 10×10のグリッド
    node_size = 10  # 各グリッドのサイズ

    # グリッド全体を描画するためのサブプロット
    fig, axes = plt.subplots(node_count, node_count, figsize=(20, 20))
    axes = axes.flatten()  # 2次元から1次元のリストに変換

    for idx, ax in enumerate(axes):
        print("id:", idx)
        i = idx % node_count
        j = idx // node_count

        x = int((i + 0.5) * node_size)
        y = int((j + 0.5) * node_size)
        print("x、y:", x, y)
        robot_position = [x, y]

        human_list = initialize_human_manager(agent_positions)
        mapinfo = makeWall(robot_position, 51)
        new_mapinfo = create_mapinfo()

        ogm = np.ones((100, 100))

        scan_and_update_map(mapinfo, new_mapinfo, robot_position, human_list, ogm)

        visible_nodes_id = extract_nodes_id(ogm, id=idx, node_count=node_count, node_dis=node_size)
        print(visible_nodes_id)

        plot_final_map(ogm, new_mapinfo, human_list, robot_position, [40, 100], ax, visible_nodes_id)

    plt.tight_layout()
    plt.savefig("combined_scan_results.png")
    plt.show()

if __name__ == "__main__":
    main()
