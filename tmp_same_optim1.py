import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gurobipy as gp
from gurobipy import GRB
from scipy.spatial import distance_matrix
from utils import one_azimuth_scan, makeWall, extract_nodes, plot_data, extract_nodes_id
from agents.human import HumanManager
import math

# 人間のマネージャーを初期化する関数
def initialize_human_manager(agent_positions):
    human_manager = HumanManager()
    human_manager.nomove(agent_positions)
    return human_manager

# マップ情報を作成する関数
def create_mapinfo():
    return [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
        [0, 0]
    ]

# ロボットのスキャンとマップの更新を行う関数
def scan_and_update_map(mapinfo, new_mapinfo, robot_position, human_list, ogm):
    ogm_info = {"size": [100, 100]}
    for destination in mapinfo:
        agent_value_list = [[agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info]
                            for agent in human_list]
        scan_data = one_azimuth_scan(ogm_info, new_mapinfo, robot_position, destination, agent_value_list, 50)
        ogm *= scan_data[0]

# 2点間の距離を計算する関数
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# カバーセットと移動経路を同時に最適化する関数
def simultaneous_optimization_test(visible_nodes_id, grid_nodes, total_nodes, time_limit=300):
    m = gp.Model("simultaneous_optimization")
    m.setParam('TimeLimit', time_limit)

    # 決定変数: x[t, i]は時刻tに都市iにいるかどうかを表す
    x = m.addVars(total_nodes, total_nodes, vtype=GRB.BINARY, name="x")  

    # 都市間の距離を計算
    distances = {(i, j): calculate_distance(grid_nodes[i], grid_nodes[j]) for i in range(total_nodes) for j in range(total_nodes) if i != j}

    m.setObjective(
        gp.quicksum(x[t, i] * x[t+1, j] * distances[i, j] for t in range(total_nodes-1) for i in range(total_nodes) for j in range(total_nodes) if i != j),
        GRB.MINIMIZE
    )

    # 各時刻 t に 1 つ以下のノードにのみいる制約
    m.addConstrs(gp.quicksum(x[t, i] for i in range(total_nodes)) <= 1 for t in range(total_nodes))

    # 連続性を保証する制約: 時刻 t に選ばれたノードは、時刻 t+1 にも必ずどこかのノードに移動する
    for t in range(total_nodes - 1):
        for i in range(total_nodes):
            m.addConstr(gp.quicksum(x[t+1, j] for j in range(total_nodes)) >= x[t, i], name=f"continuity_{t}_{i}")

    # 全てのノードがカバーされる制約
    for j in range(total_nodes):  # 0-99のすべてのノードがカバーされているかを確認
        covering_constr = []
        for i in range(total_nodes):  # 各ノード i の可視ノードを確認
            if j in visible_nodes_id[i]:
                covering_constr.append(gp.quicksum(x[t, i] for t in range(total_nodes)))
        if covering_constr:
            m.addConstr(gp.quicksum(covering_constr) >= 1, name=f"cover_{j}")

    # 最適化の途中でプロットを更新するためのコールバック関数
    def update_callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # 部分解を取得
            solution = model.cbGetSolution([x[t, i] for t in range(total_nodes) for i in range(total_nodes)])
            optimal_x = {(t, i): solution[t * total_nodes + i] for t in range(total_nodes) for i in range(total_nodes)}
            extract_and_visualize(optimal_x, grid_nodes)  # 部分解をプロット

    # 最適化の実行
    m.optimize(callback=update_callback)

    # 最適化後の最終結果
    optimal_x = {(t, i): x[t, i].X for t in range(total_nodes) for i in range(total_nodes)}
    return optimal_x

def extract_and_visualize(optimal_x, grid_nodes):
    """
    最適化されたxから1が立っている時刻とノード番号を抽出し、順番に並べて座標を可視化する。
    """
    selected_times_and_nodes = [(t, i) for (t, i), value in optimal_x.items() if value > 0.5]
    selected_times_and_nodes.sort()  # 時刻tの昇順にソート
    renumbered_times_and_nodes = [(index, grid_nodes[i]) for index, (t, i) in enumerate(selected_times_and_nodes)]

    # リアルタイムでプロットを更新
    update_plot(renumbered_times_and_nodes)

def update_plot(renumbered_times_and_nodes):
    """
    リアルタイムでプロットを更新するための関数。
    """
    x_coords = [coord[0] for _, coord in renumbered_times_and_nodes]
    y_coords = [coord[1] for _, coord in renumbered_times_and_nodes]

    plt.clf()  # クリアしてから再描画
    plt.plot(x_coords, y_coords, marker='o', color='b', linestyle='-', linewidth=2, markersize=5)

    # 各ポイントに順番を表示
    for i, (index, coord) in enumerate(renumbered_times_and_nodes):
        plt.text(coord[0], coord[1], f'{index}', fontsize=12, ha='right')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Optimized Path (Real-Time)")
    plt.grid(True)
    plt.pause(0.01)  # 短い時間で表示を更新

def main():
    """
    エージェントを初期化し、最適なカバーセットと経路を計算し可視化するメインプロセス。
    """
    agent_positions = [[70, 50, 0], [40, 50, 0], [80, 45, np.pi/4], [35, 20, np.pi/2], [65, 30, np.pi/3], [80, 30, np.pi/2]]
    node_count = 10
    node_dis = 10
    all_visible_nodes_id = {}  # 各ノードがカバーできるノードを格納する変数
    grid_nodes = {}  # ノードインデックスと座標を対応付ける辞書

    for idx in range(node_count * node_count):  # 左下から上に向かってindexを付与
        i = idx % node_count
        j = idx // node_count
        x = int((i + 0.5) * node_dis)
        y = int((j + 0.5) * node_dis)
        grid_nodes[idx] = (x, y)

        human_list = initialize_human_manager(agent_positions)
        mapinfo = makeWall([x, y], 51)  # ロボットの現在位置から壁を作成
        new_mapinfo = create_mapinfo()

        ogm = np.ones((100, 100))  # OGMを初期化
        scan_and_update_map(mapinfo, new_mapinfo, [x, y], human_list, ogm)

        visible_nodes_id = extract_nodes_id(ogm, id=idx, node_count=node_count, node_dis=node_dis)
        all_visible_nodes_id[idx] = visible_nodes_id

    # 最適化と可視化の実行
    plt.ion()  # インタラクティブモードを有効に
    optimal_path = simultaneous_optimization_test(all_visible_nodes_id, grid_nodes, total_nodes=100, time_limit=600)
    plt.ioff()  # インタラクティブモードを終了
    plt.show()

if __name__ == "__main__":
    main()