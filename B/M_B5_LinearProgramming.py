# M_B5_LinearProgramming.py

import numpy as np
from scipy.optimize import linprog

def linear_programming(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, 
                      bounds=None, method='highs'):
    """
    求解线性规划问题
    标准形式: min c^T * x
             subject to: A_ub * x <= b_ub
                        A_eq * x == b_eq
                        bounds[i][0] <= x[i] <= bounds[i][1]
    
    参数:
    c: array, 目标函数系数向量
    A_ub: array, 不等式约束系数矩阵
    b_ub: array, 不等式约束右端向量
    A_eq: array, 等式约束系数矩阵  
    b_eq: array, 等式约束右端向量
    bounds: list, 变量取值范围 [(min1,max1), (min2,max2), ...]
    method: str, 求解方法 ('highs', 'simplex')
    
    返回:
    solution: array, 最优解
    optimal_value: float, 最优目标值
    status: str, 求解状态
    """
    # 调用scipy求解器
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method=method)
    
    if result.success:
        return result.x, result.fun, 'optimal'
    else:
        return None, None, result.message

def transportation_problem(supply, demand, costs):
    """
    求解运输问题
    
    参数:
    supply: array, 各供应地的供应量
    demand: array, 各需求地的需求量  
    costs: array, 运输成本矩阵 (供应地×需求地)
    
    返回:
    solution: array, 最优运输方案矩阵
    min_cost: float, 最小运输成本
    """
    m, n = len(supply), len(demand)
    
    # 检查供需平衡
    if sum(supply) != sum(demand):
        print("警告: 供需不平衡!")
        return None, None
    
    # 构建线性规划模型
    # 决策变量: x_ij 表示从供应地i到需求地j的运输量
    c = costs.flatten()  # 目标函数系数
    
    # 等式约束: 供应约束和需求约束
    A_eq = []
    b_eq = []
    
    # 供应约束: sum(x_ij for j) = supply[i]
    for i in range(m):
        constraint = np.zeros(m * n)
        for j in range(n):
            constraint[i * n + j] = 1
        A_eq.append(constraint)
        b_eq.append(supply[i])
    
    # 需求约束: sum(x_ij for i) = demand[j]  
    for j in range(n):
        constraint = np.zeros(m * n)
        for i in range(m):
            constraint[i * n + j] = 1
        A_eq.append(constraint)
        b_eq.append(demand[j])
    
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # 变量下界约束: x_ij >= 0
    bounds = [(0, None) for _ in range(m * n)]
    
    # 求解
    solution, min_cost, status = linear_programming(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    
    if status == 'optimal':
        # 重新整理为矩阵形式
        solution_matrix = solution.reshape(m, n)
        return solution_matrix, min_cost
    else:
        return None, None

def diet_problem(nutrients, foods, requirements, costs):
    """
    求解饮食问题 (营养搭配优化)
    
    参数:
    nutrients: list, 营养素名称列表
    foods: list, 食物名称列表
    requirements: array, 各营养素的最低需求量
    costs: array, 各食物的单价
    
    返回:
    solution: array, 最优食物搭配量
    min_cost: float, 最小总成本
    """
    # 营养素含量矩阵 (营养素×食物)
    # 这里需要根据实际问题提供数据
    # 示例: 假设有蛋白质、脂肪、碳水化合物三种营养素，米饭、肉类、蔬菜三种食物
    nutrition_matrix = np.array([
        [2.5, 20.0, 2.0],   # 蛋白质含量 (g/100g)
        [0.5, 15.0, 0.2],   # 脂肪含量
        [80.0, 0.0, 8.0]    # 碳水化合物含量
    ])
    
    # 约束: 营养素摄入量 >= 最低需求
    A_ub = -nutrition_matrix.T  # 转置并取负号 (因为要求 >= 转为 <= )
    b_ub = -requirements        # 取负号
    
    # 变量下界: 食物量 >= 0
    bounds = [(0, None) for _ in range(len(foods))]
    
    # 求解
    solution, min_cost, status = linear_programming(costs, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    return solution, min_cost if status == 'optimal' else (None, None)

if __name__ == '__main__':
    # --- 使用示例: 生产计划问题 ---
    print("示例1: 生产计划问题")
    print("某工厂生产两种产品A和B，求最大利润的生产计划")
    
    # 目标: 最大化利润 (转为最小化负利润)
    # 产品A利润40元/件，产品B利润30元/件
    c = [-40, -30]  # 最小化 -40*x1 - 30*x2
    
    # 约束条件:
    # 原料约束: 2*x1 + x2 <= 100
    # 工时约束: x1 + 2*x2 <= 80  
    # 非负约束: x1, x2 >= 0
    A_ub = [[2, 1], [1, 2]]
    b_ub = [100, 80]
    bounds = [(0, None), (0, None)]
    
    solution, optimal_value, status = linear_programming(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    if status == 'optimal':
        print(f"最优解: 产品A={solution[0]:.2f}件, 产品B={solution[1]:.2f}件")
        print(f"最大利润: {-optimal_value:.2f}元")  # 取负号还原
    
    # --- 示例2: 运输问题 ---
    print("\n示例2: 运输问题")
    print("3个供应地向3个需求地运输货物，求最小运输成本")
    
    supply = [20, 30, 25]     # 供应量
    demand = [25, 20, 30]     # 需求量
    costs = np.array([        # 运输成本矩阵
        [8, 6, 10],
        [9, 12, 13], 
        [14, 9, 16]
    ])
    
    transport_plan, min_transport_cost = transportation_problem(supply, demand, costs)
    
    if transport_plan is not None:
        print("最优运输方案:")
        for i in range(len(supply)):
            for j in range(len(demand)):
                if transport_plan[i,j] > 0.01:  # 只显示非零运输量
                    print(f"  供应地{i+1} -> 需求地{j+1}: {transport_plan[i,j]:.1f}")
        print(f"最小运输成本: {min_transport_cost:.2f}")
    
    # 如何修改为你自己的问题:
    # 1. 确定决策变量，构建目标函数系数向量c
    # 2. 列出约束条件，构建约束矩阵A和约束向量b  
    # 3. 设定变量取值范围bounds
    # 4. 调用linear_programming()求解
    # 5. 检查求解状态，解释最优解的实际意义