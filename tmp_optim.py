import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from scipy.spatial import distance_matrix
from utils import one_azimuth_scan, makeWall, extract_nodes, plot_data, extract_nodes_id
from agents.human import HumanManager
import math

# 人間のマネージャーを初期化する関数
def initialize_human_manager(agent_positions):
    """
    HumanManagerオブジェクトを初期化し、指定されたエージェントの位置に基づいて移動を無効化。
    
    Parameters:
    - agent_positions: エージェントの初期位置のリスト
    
    Returns:
    - 初期化されたHumanManagerオブジェクト
    """
    human_manager = HumanManager()
    human_manager.nomove(agent_positions)
    return human_manager

# マップ情報を作成する関数
def create_mapinfo():
    """
    ロボットが動作するための簡易な地図情報を返す。
    
    Returns:
    - 地図情報を含むリスト（矩形の壁を示す座標のリスト）
    """
    return [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
        [0, 0]
    ]

# ロボットのスキャンとマップの更新を行う関数
def scan_and_update_map(mapinfo, new_mapinfo, robot_position, human_list, ogm):
    """
    ロボットのスキャン情報を基に地図を更新。
    
    Parameters:
    - mapinfo: 現在のマップ情報
    - new_mapinfo: 新しいマップ情報
    - robot_position: ロボットの現在の位置
    - human_list: 人間エージェントのリスト
    - ogm: Occupancy Grid Map（占有グリッドマップ）
    """
    ogm_info = {"size": [100, 100]}
    for destination in mapinfo:
        agent_value_list = [[agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info]
                            for agent in human_list]
        scan_data = one_azimuth_scan(ogm_info, new_mapinfo, robot_position, destination, agent_value_list, 50)
        ogm *= scan_data[0]

# カバーセット問題を解く関数
def optimize_cover_set(visible_nodes, total_nodes):
    """
    Gurobiを使用して、最小のカバーセット問題を解く。
    
    Parameters:
    - visible_nodes: 各ノードがカバーできるノードのリスト
    - total_nodes: 合計ノード数
    
    Returns:
    - 最適なカバーセットのノードインデックスのリスト
    """
    m = gp.Model("cover_set")
    num_nodes = len(visible_nodes)
    y = m.addVars(num_nodes, vtype=GRB.BINARY, name="y")

    for j in range(total_nodes):
        covering_nodes = [i for i in range(num_nodes) if j in visible_nodes[i]]
        if covering_nodes:
            m.addConstr(gp.quicksum(y[i] for i in covering_nodes) >= 1, name=f"cover_{j}")
        else:
            print(f"Warning: ノード {j} はカバーできるノードがありません。")

    m.setObjective(gp.quicksum(y[i] for i in range(num_nodes)), GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.INFEASIBLE:
        print("モデルは実行不可能です。IISを生成します。")
        m.computeIIS()
        m.write("infeasible_model.ilp")
        return []
    
    covering_set = [i for i in range(num_nodes) if y[i].X > 0.5]
    return covering_set

# 2点間の距離を計算する関数
def calculate_distance(point1, point2):
    """
    2つの点のユークリッド距離を計算。
    
    Parameters:
    - point1: 点1の座標 (x, y)
    - point2: 点2の座標 (x, y)
    
    Returns:
    - 点1と点2の間の距離
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def plot_final_map(ogm, new_mapinfo, node_coordinates, human_list, robot_position, destination, ax):
    """最終的なOGMとエージェント情報をプロットする"""
    plot_data(ogm, None, ax, node_coordinates, mapinfo=new_mapinfo, agent_list=human_list, dst=destination, now=robot_position, axis_option=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

# Gurobiを使ってハミルトンパス問題を解く関数（スタートに戻らない最短経路）
def optimize_shortest_hamiltonian_path_gurobi(coordinates):
    """
    Gurobiを使用して、戻らないハミルトンパス（最短経路）を解きます。
    
    Parameters:
    - coordinates: 各ノードの座標のリスト
    
    Returns:
    - 最適なハミルトンパスのノードの順序リスト
    """
    n = len(coordinates)
    
    # Gurobiモデルの作成
    m = gp.Model()

    # 都市間の距離を計算
    distances = {(i, j): calculate_distance(coordinates[i], coordinates[j]) for i in range(n) for j in range(n) if i != j}

    # 決定変数: x[t, i]は時刻tに都市iにいるかどうかを表す
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")

    # 各時刻tに1つの都市にのみいる制約
    m.addConstrs(gp.quicksum(x[t, i] for i in range(n)) == 1 for t in range(n))
    
    # 各都市iには1回だけ訪れる制約
    m.addConstrs(gp.quicksum(x[t, i] for t in range(n)) == 1 for i in range(n))

    # 目的関数：時刻tからt+1に移動する距離を最小化
    m.setObjective(
        gp.quicksum(x[t, i] * x[t+1, j] * distances[i, j] for t in range(n-1) for i in range(n) for j in range(n) if i != j),
        GRB.MINIMIZE
    )

    # 最適化を実行
    m.optimize()

    # 最適な経路を抽出
    if m.status == GRB.OPTIMAL:
        solution = m.getAttr('x', x)
        path = []
        visited = set()

        # 最適経路を時系列順に追加
        for t in range(n):
            for i in range(n):
                if solution[t, i] > 0.5:
                    path.append(i)
                    visited.add(i)
                    break

        return path
    else:
        return None

    
# 結果を可視化する関数（エッジを追加）
def visualize_shortest_path(coordinates, path):
    """
    最適化されたハミルトンパスの結果を座標間にエッジを引いて可視化します。
    
    Parameters:
    - coordinates: ノードの座標リスト
    - path: 最適なノードの順序リスト（時刻ごとの都市の順番）
    """
    x_coords = [coordinates[i][0] for i in path]
    y_coords = [coordinates[i][1] for i in path]

    plt.figure(figsize=(8, 6))
    
    plt.plot(x_coords, y_coords, marker='o', color='b', linestyle='-', linewidth=2, markersize=5)

    for i, coord in enumerate(path):
        plt.text(coordinates[coord][0], coordinates[coord][1], f'{i}', fontsize=12, ha='right')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.title("Optimized Shortest Hamiltonian Path (No Return)")
    plt.grid(True)
    plt.show()

# covering_setが全てのノードをカバーしているかを確認するための関数
def check_covering_set(all_visible_nodes, covering_set, total_nodes=100):
    """
    covering_setによって、0からtotal_nodes-1までの全てのノードがカバーされているかを確認する。

    Parameters:
    - all_visible_nodes: 各ノードがカバーできるノードのリスト
    - covering_set: 最適なカバーセットのノードインデックスのリスト
    - total_nodes: 合計ノード数 (デフォルトは100)

    Returns:
    - True: 全てのノードがカバーされている場合
    - False: 1つでもカバーされていないノードがある場合
    """
    check = set()

    # covering_setに含まれるノードがカバーする全てのノードを`check`に追加
    for idx in covering_set:
        check.update(all_visible_nodes[idx])

    # checkリストに0からtotal_nodes-1までの全ノードが含まれているか確認
    print("全てのノード:", check)
    print("ノードの長さ:", len(check))
    missing_nodes = set(range(total_nodes)) - check
    if missing_nodes:
        print(f"カバーされていないノードがあります: {missing_nodes}")
    else:
        print("全てのノードがカバーされています。")

# メイン関数
def main():
    """
    人間エージェントを初期化し、最適なカバーセットとハミルトンパスを計算して結果を可視化するメインプロセス。
    """
    agent_positions = [[70, 50, 0], [40, 50, 0], [80, 45, np.pi/4], [35, 20, np.pi/2], [65, 30, np.pi/3], [80, 30, np.pi/2]]
    node_count = 10
    node_size = 10
    fig, axes = plt.subplots(node_count, node_count, figsize=(20, 20))
    axes = axes.flatten()
    all_visible_nodes = {}
    grid_nodes = []

    for idx, ax in enumerate(axes):
        i = idx // node_count
        j = idx % node_count
        x = int((i + 0.5) * node_size)
        y = int((j + 0.5) * node_size)
        robot_position = [x, y]
        grid_nodes.append((x, y))

        human_list = initialize_human_manager(agent_positions)
        mapinfo = makeWall(robot_position, 51)
        new_mapinfo = create_mapinfo()
        ogm = np.ones((100, 100))

        scan_and_update_map(mapinfo, new_mapinfo, robot_position, human_list, ogm)

        current_node = (x, y)
        visible_nodes = extract_nodes(ogm, current_node=current_node, node_count=10)
        visible_nodes_id = extract_nodes_id(ogm, current_node=current_node, node_count=10)
        all_visible_nodes[idx] = visible_nodes_id

        plot_final_map(ogm, new_mapinfo, visible_nodes, human_list, robot_position, [40, 100], ax)

    plt.tight_layout()
    plt.savefig("combined_scan_results.png")
    plt.show()

    covering_set = optimize_cover_set(all_visible_nodes, len(grid_nodes))
    check_covering_set(all_visible_nodes, covering_set, total_nodes=100)
    selected_coordinates = [grid_nodes[i] for i in covering_set]
    print("Best Nodes:", selected_coordinates)
    optimal_path = optimize_shortest_hamiltonian_path_gurobi(selected_coordinates)

    if optimal_path:
        print("Optimal Path:", optimal_path)
        visualize_shortest_path(selected_coordinates, optimal_path)
    else:
        print("Optimal solution not found.")
    
if __name__ == "__main__":
    main()
