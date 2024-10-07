from code.class_MapManager import MapManager
from code.class_Optimizer import Optimizer
from code.visualizer import extract_and_visualize, plot_vis
from agents.human import HumanManager
import numpy as np
import matplotlib.pyplot as plt
from utils import makeWall

AGENT_POSITIONS = [[70, 50, 0], [40, 50, 0], [80, 45, np.pi/4], [35, 20, np.pi/2], [65, 30, np.pi/3], [80, 30, np.pi/2]]
NODE_GRID = 17 #[1m]
NODE_INTERVAL = 6 #[1m]
OPTIM_TIME = 600 * 30 #[1200s]
MAX_NODE = 20
WEIGHT = 10
INIT_ROBOT = None

def main():
    map_manager = MapManager(AGENT_POSITIONS, node_count=NODE_GRID, node_dis=NODE_INTERVAL)
    grid_nodes = map_manager.create_grid() #ノードインデックスと座標を対応付ける辞書

    # 各ノードに対してスキャンと可視ノード抽出を実行
    for idx, (x, y) in grid_nodes.items():
        print(f"Index: {idx}, X: {x}, Y: {y}")
        robot_position = [x, y]
        ogm = map_manager.scan_environment(robot_position)
        visible_nodes_id = map_manager.extract_visible_nodes_id(ogm, idx)
        map_manager.all_visible_nodes_id[idx] = visible_nodes_id
        if visible_nodes_id:
            print(f"  Visible Nodes from Node {idx}: {', '.join(map(str, visible_nodes_id))}")
        else:
            print(f"  No visible nodes from Node {idx}.")
    
    # 最適化計算
    optimizer = Optimizer(time_limit=OPTIM_TIME)
    optimal_path = optimizer.simultaneous_optimization(map_manager.all_visible_nodes_id, grid_nodes, total_nodes=len(grid_nodes), max_nodes=MAX_NODE, weight=WEIGHT, initial_id=INIT_ROBOT)
    for t, i in optimal_path:
        print(f"Time: {t}, Node id: {i}, pos: {grid_nodes[i]}")

    # 可視化
    sorted_pos= extract_and_visualize(optimal_path, grid_nodes)
    print(f"Optimized Pos: {sorted_pos}")

    ogm = np.ones((100, 100))
    for pos in sorted_pos:
        mapinfo = makeWall(pos, 51)
        new_mapinfo = map_manager.new_mapinfo
        map_manager.scan_and_update_map(mapinfo, new_mapinfo, pos, ogm)

    plot_vis(data=ogm, grid_nodes=grid_nodes, pos=sorted_pos, agent_list=map_manager.human_manager, mapinfo=new_mapinfo)

if __name__ == "__main__":
    main()
