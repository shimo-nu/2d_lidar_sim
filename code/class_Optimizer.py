import gurobipy as gp
from gurobipy import GRB
import math

class Optimizer:
    def __init__(self, time_limit=300):
        self.time_limit = time_limit

    @staticmethod
    def calculate_distance(point1, point2):
        """2点間の距離を小数点以下3桁まで計算"""
        return round(math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2), 2)

    def simultaneous_optimization(self, visible_nodes_id, grid_nodes, total_nodes, max_nodes, weight, initial_id=None):
        """Gurobiを使った最適化処理"""

        print("Create OptimModel")
        print("total node:", total_nodes)
        m = gp.Model("simultaneous_optimization")
        m.setParam('TimeLimit', self.time_limit)

        # 決定変数: x[t, i]は時刻tにノードiにいるかどうか
        x = m.addVars(total_nodes, total_nodes, vtype=GRB.BINARY, name="x")

        # 距離計算
        distances = {(i, j): self.calculate_distance(grid_nodes[i], grid_nodes[j]) for i in range(total_nodes) for j in range(total_nodes) if i != j}

        # 移動距離の最小化の目的関数の部分
        objective_distance = gp.quicksum(x[t, i] * x[t + 1, j] * distances[i, j] 
                                        for t in range(total_nodes - 1) 
                                        for i in range(total_nodes) 
                                        for j in range(total_nodes) if i != j)

        # ノード数を少なくするための目的関数の項
        objective_nodes = gp.quicksum(x[t, i] for t in range(total_nodes) for i in range(total_nodes))

        # 目的関数に両方の項を組み込む
        # m.setObjective(objective_distance + weight * objective_nodes, GRB.MINIMIZE)
        m.setObjective(objective_distance, GRB.MINIMIZE)

        # 制約条件: 各時刻 t に 1 つ以下のノードにのみいる
        m.addConstrs(gp.quicksum(x[t, i] for i in range(total_nodes)) <= 1 for t in range(total_nodes))
        m.addConstrs(gp.quicksum(x[t, i] for t in range(total_nodes)) <= 1 for i in range(total_nodes))

        # 連続性の制約
        for t in range(total_nodes - 1):
            for i in range(total_nodes):
                m.addConstr(gp.quicksum(x[t+1, j] for j in range(total_nodes)) >= x[t, i], name=f"continuity_{t}_{i}")

        # カバーセット制約
        for j in range(total_nodes):
            covering_constr = [gp.quicksum(x[t, i] for t in range(total_nodes)) for i in range(total_nodes) if j in visible_nodes_id[i]]
            if covering_constr:
                m.addConstr(gp.quicksum(covering_constr) >= 1, name=f"cover_{j}")

        # 全体で訪問するノード数を制限する制約
        m.addConstr(
            gp.quicksum(x[t, i] for t in range(total_nodes) for i in range(total_nodes)) <= max_nodes, 
            name="max_total_nodes"
        )

        # 初期位置の制約: initial_id が指定されている場合に限り制約を追加
        if initial_id is not None:
            print(f"Initial ID exists: {initial_id}, position: {grid_nodes[initial_id]}")

            first_visit = m.addVars(total_nodes, vtype=GRB.BINARY, name="first_visit")

            # 強制的に1が立つようにする
            m.addConstr(gp.quicksum(first_visit[t] for t in range(total_nodes)) == 1, name="unique_first_visit")

            # 初めて訪れる時刻に1が立つことを保証
            for t in range(total_nodes):
                m.addConstr(x[t, initial_id] == first_visit[t], name=f"first_visit_{t}")

            # first_visit の時刻よりも前のすべての時刻で、どのノードも訪問しない（すべて0）
            for t in range(total_nodes):
                m.addConstr(
                    gp.quicksum(x[tt, i] for tt in range(t) for i in range(total_nodes)) <= (1 - first_visit[t]) * total_nodes,
                    name=f"no_visit_before_first_visit_{t}"
            )   
                
        print("Finish Create OptimModel")
        print("Start Optimize!")
        m.optimize()

        # 最適解を取得
        optimal_positions = [(t, i) for t in range(total_nodes) for i in range(total_nodes) if x[t, i].X > 0.5]

        return optimal_positions
