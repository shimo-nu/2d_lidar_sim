import numpy as np
from utils import one_azimuth_scan, makeWall
from agents.human import HumanManager

class MapManager:
    def __init__(self, agent_positions, node_count=10, node_dis=10):
        """
        コンストラクタで必要な情報を初期化します。
        agent_positions: エージェントの初期位置
        node_count: グリッドのノード数
        node_dis: ノード間の距離
        """
        self.agent_positions = agent_positions
        self.node_count = node_count
        self.node_dis = node_dis
        self.grid_nodes = {}  # グリッドのノード座標
        self.all_visible_nodes_id = {}  # 全ノードの可視ノードID

        self.human_manager = HumanManager()
        self.human_manager.nomove(self.agent_positions)

        self.new_mapinfo = self.create_mapinfo()

    def create_grid(self):
        """
        グリッドを作成し、各ノードの座標を格納します。
        戻り値: ノードの座標を格納した辞書
        """
        for idx in range(self.node_count * self.node_count):
            i = idx % self.node_count
            j = idx // self.node_count
            x = int((i + 0.5) * self.node_dis)
            y = int((j + 0.5) * self.node_dis)
            self.grid_nodes[idx] = (x, y)
        return self.grid_nodes

    def scan_environment(self, robot_position):
        """
        スキャンデータを生成し、可視ノードを抽出します。
        robot_position: ロボットの位置
        human_list: 人間エージェントのリスト
        戻り値: スキャン結果（OGM）
        """
        ogm = np.ones((100, 100))
        mapinfo = makeWall(robot_position, 51)

        # スキャンと更新
        self.scan_and_update_map(mapinfo, self.new_mapinfo, robot_position, ogm)
        return ogm

    def extract_visible_nodes_id(self, data, id=None):
        """
        Occupancy Grid Map（OGM）から可視ノードのIDを抽出します。
        data: Occupancy Grid Map
        id: 現在のノードID
        戻り値: 可視ノードのIDリスト
        """
        node_indices = []

        for idx, (x, y) in self.grid_nodes.items():
            ogm_x = int(x)
            ogm_y = int(y)

            if data[ogm_x, ogm_y] == 0:
                node_indices.append(idx)

        if id is not None:
            node_indices.append(id)

        node_indices.sort()
        return node_indices

    def create_mapinfo(self):
        """
        地図の基本情報を返します。
        戻り値: 地図情報のリスト
        """
        return [
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100],
            [0, 0]
        ]

    def scan_and_update_map(self, mapinfo, new_mapinfo, robot_position, ogm):
        """
        ロボットのスキャン情報を基に地図を更新します。
        """
        ogm_info = {"size": [100, 100]}
        for destination in mapinfo:
            agent_value_list = [
                [agent.isArrived, agent.isStarted, agent.position, agent.size, agent.unique_num, agent.agent_info]
                for agent in self.human_manager]
            scan_data = one_azimuth_scan(ogm_info, new_mapinfo, robot_position, destination, agent_value_list, 50)
            ogm *= scan_data[0]

