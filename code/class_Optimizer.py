import gurobipy as gp
from gurobipy import GRB
import math
import json
import os
import matplotlib.pyplot as plt

class Optimizer:
    def __init__(self, time_limit=300):
        self.time_limit = time_limit
        self.log_data = []  # ログデータを保存するリスト

    @staticmethod
    def calculate_distance(point1, point2):
        """2点間の距離を小数点以下3桁まで計算"""
        return round(math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2), 2)

    @staticmethod
    def get_next_log_index(directory="outputs", prefix="solution", extension=".json"):
        """ディレクトリ内の既存ファイル数に基づき、次のファイルインデックスを取得"""
        if not os.path.exists(directory):
            return 0  # ディレクトリが存在しなければ0を返す
        existing_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
        return len(existing_files)

    def save_log_to_json(self):
        """ログをJSONファイルに保存する"""
        os.makedirs("outputs", exist_ok=True)
        index = self.get_next_log_index("outputs", "solution", ".json")
        file_path = f"outputs/solution{index}.json"
        with open(file_path, "w") as f:
            json.dump(self.log_data, f, indent=4)
        print(f"Log saved to {file_path}")

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

        # 目的関数に両方の項を組み込む
        m.setObjective(objective_distance, GRB.MINIMIZE)

        # 制約条件
        m.addConstrs(gp.quicksum(x[t, i] for i in range(total_nodes)) <= 1 for t in range(total_nodes))
        m.addConstrs(gp.quicksum(x[t, i] for t in range(total_nodes)) <= 1 for i in range(total_nodes))

        for t in range(total_nodes - 1):
            for i in range(total_nodes):
                m.addConstr(gp.quicksum(x[t+1, j] for j in range(total_nodes)) >= x[t, i], name=f"continuity_{t}_{i}")
                
        for j in range(total_nodes):
            covering_constr = [gp.quicksum(x[t, i] for t in range(total_nodes)) for i in range(total_nodes) if j in visible_nodes_id[i]]
            if covering_constr:
                m.addConstr(gp.quicksum(covering_constr) >= 1, name=f"cover_{j}")
        
        m.addConstr(
            gp.quicksum(x[t, i] for t in range(total_nodes) for i in range(total_nodes)) <= max_nodes, 
            name="max_total_nodes"
        )
        if initial_id is not None:
            first_visit = m.addVars(total_nodes, vtype=GRB.BINARY, name="first_visit")
            m.addConstr(gp.quicksum(first_visit[t] for t in range(total_nodes)) == 1, name="unique_first_visit")
            for t in range(total_nodes):
                m.addConstr(x[t, initial_id] == first_visit[t], name=f"first_visit_{t}")
            for t in range(total_nodes):
                m.addConstr(
                    gp.quicksum(x[tt, i] for tt in range(t) for i in range(total_nodes)) <= (1 - first_visit[t]) * total_nodes,
                    name=f"no_visit_before_first_visit_{t}"
            )   
                
        print("Finish Create OptimModel")
        print("Start Optimize!")

        def log_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    runtime = model.cbGet(GRB.Callback.RUNTIME)
                    objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                    objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                    node_count = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
                    sol_count = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)

                    # x の値が 1 の場合のみ保存
                    x_values = {f"x[{t},{i}]": model.cbGetSolution(x[t, i]) 
                                for t in range(total_nodes) 
                                for i in range(total_nodes) 
                                if model.cbGetSolution(x[t, i]) > 0.5}  # 1に近いものだけ

                    # ログデータに保存
                    self.log_data.append({
                        "type": "MIPSOL",
                        "objective": obj_value,
                        "runtime": runtime,
                        "objective_best": objbst,
                        "objective_bound": objbnd,
                        "node_count": node_count,
                        "solution_count": sol_count,
                        "x_values": x_values  # 1 のみ保存
                    })
                except gp.GurobiError as e:
                    print(f"Error in MIPSOL callback: {e}")

        m.optimize(log_callback)

        optimal_positions = [(t, i) for t in range(total_nodes) for i in range(total_nodes) if x[t, i].X > 0.5]
        
        self.save_log_to_json()

        return optimal_positions

    def visualize_optimization_log(self):
        """最適化プロセス中の目的関数の変化をプロットし、画像を保存"""
        runtimes = [entry["runtime"] for entry in self.log_data]
        objectives = [entry["objective"] for entry in self.log_data]

        plt.figure()
        plt.plot(runtimes, objectives, marker='o', linestyle='-')
        plt.xlabel("Runtime (seconds)")
        plt.ylabel("Objective Value")
        plt.title("Objective Value Over Time")
        plt.grid(True)
        os.makedirs("image", exist_ok=True)
        index = self.get_next_log_index("image", "optim_log", ".png")
        file_path = f"image/optim_log{index}.png"
        plt.savefig(file_path)
        print(f"Visualization saved to {file_path}")

