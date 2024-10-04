import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import MaxNLocator

def extract_and_visualize(optimal_path, grid_nodes):
    """
    最適化されたoptimal_pathからノードidと座標を抽出し、時系列で可視化する。
    
    Parameters:
    - optimal_path: 最適化された経路（時刻とノードのペア）
    - grid_nodes: ノードインデックスと座標の対応辞書
    """
    # 時刻順に座標を格納
    sorted_coordinates = []

    # optimal_path の要素でノードiに基づいてループ
    for idx, (_, i) in enumerate(optimal_path):
        node_pos = grid_nodes[i]  # ノードの座標を取得
        sorted_coordinates.append(node_pos)
        print(f"Time: {idx}, Node id: {i}, Position: {node_pos}")

    # 座標リストを可視化
    visualize_path(sorted_coordinates)

    # 座標リストを返す
    return sorted_coordinates

def visualize_path(sorted_coordinates):
    """
    経路の座標を可視化するメソッド。
    
    Parameters:
    - sorted_coordinates: 時系列順に並んだノードの座標リスト
    """
    x_coords = [coord[0] for coord in sorted_coordinates]
    y_coords = [coord[1] for coord in sorted_coordinates]

    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='o', color='b', linestyle='-', linewidth=2, markersize=5)

    # 各ポイントに順番を表示
    for idx, (x, y) in enumerate(sorted_coordinates):
        plt.text(x, y, f'{idx}', fontsize=12, ha='right')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Optimized Path")
    plt.grid(True)
    plt.savefig("image/optim_path.png")
    plt.show()
    
def plot_vis(data, grid_nodes, pos, agent_list=None, mapinfo=None, title="" , axis_option=False):
    fig, ax = plt.subplots()
    data_size = data.shape

    data_plot = ax.imshow(data, cmap="Purples", vmin=0, vmax=1, origin='upper', 
                          extent=[0, data_size[0], data_size[1], 0], interpolation='none', alpha=1)

    if agent_list is not None:
        for agent in agent_list:
            ax.plot(agent.position[0], agent.position[1], c="#42F371", marker=".", markersize=5)

    for idx, (x, y) in grid_nodes.items():
        ax.plot(x, y, 'o', c="green", markersize=2)

    ax.set_title(title)
    
    if pos is not None:
        for robot_pos in pos:
            ax.plot(robot_pos[0], robot_pos[1], '.', c="blue", markersize=10)
    
    if mapinfo is not None:
        x = [i[0] for i in mapinfo]
        y = [i[1] for i in mapinfo]
        ax.plot(x, y)
    
    if axis_option and fig is not None:
        fig.colorbar(data_plot)

    # 軸の設定
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("image/scan_results.png")
    plt.show()