import gurobipy as gp
from gurobipy import GRB
import math

class Optimizer:
    def __init__(self, time_limit=300):
        self.time_limit = time_limit

    @staticmethod
    def calculate_distance(point1, point2):
        """2点間の距離を計算"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def simultaneous_optimization(self, visible_nodes_id, grid_nodes, total_nodes):
        """Gurobiを使った最適化処理"""
        m = gp.Model("simultaneous_optimization")
        m.setParam('TimeLimit', self.time_limit)

        # 決定変数: x[t, i]は時刻tにノードiにいるかどうか
        x = m.addVars(total_nodes, total_nodes, vtype=GRB.BINARY, name="x")

        # 距離計算
        distances = {(i, j): self.calculate_distance(grid_nodes[i], grid_nodes[j]) for i in range(total_nodes) for j in range(total_nodes) if i != j}

        # 目的関数: 移動距離の最小化
        m.setObjective(
            gp.quicksum(x[t, i] * x[t+1, j] * distances[i, j] for t in range(total_nodes-1) for i in range(total_nodes) for j in range(total_nodes) if i != j),
            GRB.MINIMIZE
        )

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

        m.optimize()

        # 最適解を取得
        optimal_positions = [(t, i) for t in range(total_nodes) for i in range(total_nodes) if x[t, i].X > 0.5]

        return optimal_positions
