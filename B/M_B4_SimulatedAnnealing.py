# M_B4_SimulatedAnnealing.py

import numpy as np
import random

def simulated_annealing(objective_func, bounds, initial_temp=1000, cooling_rate=0.95, 
                       min_temp=1e-6, max_iter=10000, minimize=True):
    """
    使用模拟退火算法进行全局优化
    
    参数:
    objective_func: function, 目标函数 f(x)
    bounds: list, 变量取值范围 [[min1,max1], [min2,max2], ...]
    initial_temp: float, 初始温度
    cooling_rate: float, 降温率 (0 < cooling_rate < 1)
    min_temp: float, 最低温度
    max_iter: int, 最大迭代次数
    minimize: bool, True为最小化，False为最大化
    
    返回:
    best_solution: array, 最优解
    best_fitness: float, 最优值
    history: list, 迭代历史
    """
    # 初始化
    bounds = np.array(bounds)
    dimension = len(bounds)
    
    # 随机初始解
    current_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], dimension)
    current_fitness = objective_func(current_solution)
    
    # 最优解记录
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # 迭代历史
    history = []
    temperature = initial_temp
    
    for iteration in range(max_iter):
        if temperature < min_temp:
            break
            
        # 生成邻域解 (高斯扰动)
        perturbation = np.random.normal(0, temperature/initial_temp, dimension)
        new_solution = current_solution + perturbation
        
        # 边界处理 (反弹)
        for i in range(dimension):
            if new_solution[i] < bounds[i, 0]:
                new_solution[i] = bounds[i, 0] + (bounds[i, 0] - new_solution[i])
            elif new_solution[i] > bounds[i, 1]:
                new_solution[i] = bounds[i, 1] - (new_solution[i] - bounds[i, 1])
            # 再次检查边界
            new_solution[i] = np.clip(new_solution[i], bounds[i, 0], bounds[i, 1])
        
        # 计算新解适应度
        new_fitness = objective_func(new_solution)
        
        # 接受准则判断
        if minimize:
            delta = new_fitness - current_fitness
        else:
            delta = current_fitness - new_fitness
            
        if delta < 0 or random.random() < np.exp(-delta / temperature):
            current_solution = new_solution
            current_fitness = new_fitness
            
            # 更新最优解
            if (minimize and current_fitness < best_fitness) or \
               (not minimize and current_fitness > best_fitness):
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        # 记录历史
        history.append({
            'iteration': iteration,
            'temperature': temperature,
            'current_fitness': current_fitness,
            'best_fitness': best_fitness
        })
        
        # 降温
        temperature *= cooling_rate
    
    return best_solution, best_fitness, history

def sa_tsp(distance_matrix):
    """
    使用模拟退火算法求解旅行商问题(TSP)
    
    参数:
    distance_matrix: ndarray, 距离矩阵
    
    返回:
    best_path: list, 最优路径
    best_distance: float, 最优距离
    """
    n_cities = len(distance_matrix)
    
    def tsp_objective(path):
        """计算路径总距离"""
        total_distance = 0
        for i in range(len(path)):
            total_distance += distance_matrix[int(path[i])][int(path[(i + 1) % len(path)])]
        return total_distance
    
    def generate_neighbor(path):
        """生成邻域解 - 2-opt交换"""
        new_path = path.copy()
        i, j = sorted(random.sample(range(len(path)), 2))
        new_path[i:j+1] = reversed(new_path[i:j+1])
        return new_path
    
    # 初始化随机路径
    current_path = list(range(n_cities))
    random.shuffle(current_path)
    current_distance = tsp_objective(current_path)
    
    best_path = current_path.copy()
    best_distance = current_distance
    
    # 模拟退火参数
    temperature = 1000
    cooling_rate = 0.995
    min_temp = 1e-3
    
    while temperature > min_temp:
        for _ in range(100):  # 在每个温度下进行多次尝试
            new_path = generate_neighbor(current_path)
            new_distance = tsp_objective(new_path)
            
            delta = new_distance - current_distance
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_path = new_path
                current_distance = new_distance
                
                if current_distance < best_distance:
                    best_path = current_path.copy()
                    best_distance = current_distance
        
        temperature *= cooling_rate
    
    return best_path, best_distance

if __name__ == '__main__':
    # --- 使用示例: 函数优化问题 ---
    
    # 示例1: Rastrigin函数优化 (多峰函数)
    print("示例1: Rastrigin函数优化")
    def rastrigin(x):
        A = 10
        n = len(x)
        return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)
    
    bounds = [[-5.12, 5.12], [-5.12, 5.12]]  # 2维问题
    
    best_sol, best_val, history = simulated_annealing(
        rastrigin, bounds, initial_temp=100, max_iter=5000
    )
    
    print(f"最优解: [{best_sol[0]:.4f}, {best_sol[1]:.4f}]")
    print(f"最优值: {best_val:.4f}")
    print(f"理论最优值: 0 (在x=[0,0]处)")
    
    # 示例2: 简单TSP问题
    print("\n示例2: 旅行商问题(TSP)")
    # 5个城市的距离矩阵
    distance_matrix = np.array([
        [0,  29, 20, 21, 16],
        [29, 0,  15, 29, 28], 
        [20, 15, 0,  15, 14],
        [21, 29, 15, 0,  4 ],
        [16, 28, 14, 4,  0 ]
    ])
    
    best_path, best_distance = sa_tsp(distance_matrix)
    
    print(f"最优路径: {' -> '.join(map(str, best_path))} -> {best_path[0]}")
    print(f"最优距离: {best_distance}")
    
    # 如何修改为你自己的问题:
    # 1. 对于连续优化问题，定义目标函数和变量边界，调用 simulated_annealing()
    # 2. 对于组合优化问题，需要自定义邻域生成函数和目标函数
    # 3. 调整温度参数：初始温度要够高，降温率要合适 (0.9-0.99)
    # 4. 增加迭代次数可以提高解的质量，但会增加计算时间