# M_B3_ParticleSwarmOptimization.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def particle_swarm_optimization(objective_func, bounds, n_particles, n_iter, w=0.5, c1=2.0, c2=2.0, minimize=True):
    """
    使用粒子群算法求解函数优化问题
    
    参数:
    objective_func: function, 目标函数 f(x)
    bounds: array, 变量的取值范围, e.g., [[min1, max1], [min2, max2], ...]
    n_particles: int, 粒子数量
    n_iter: int, 迭代次数
    w: float, 惯性权重
    c1: float, 个体学习因子
    c2: float, 社会学习因子
    minimize: bool, True为最小化，False为最大化
    
    返回:
    best_position: array, 找到的最优解
    best_fitness: float, 最优解对应的函数值
    fitness_history: list, 每代最优适应度历史
    """
    
    n_dims = len(bounds)
    bounds = np.array(bounds)
    
    # 初始化粒子位置和速度
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_particles, n_dims))
    velocities = np.random.uniform(-1, 1, (n_particles, n_dims))
    
    # 初始化个体最优位置和全局最优位置
    personal_best_positions = positions.copy()
    personal_best_fitness = np.array([objective_func(pos) for pos in positions])
    
    if minimize:
        global_best_idx = np.argmin(personal_best_fitness)
    else:
        global_best_idx = np.argmax(personal_best_fitness)
        
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_fitness = personal_best_fitness[global_best_idx]
    
    fitness_history = [global_best_fitness]
    
    # 主循环
    for iteration in range(n_iter):
        for i in range(n_particles):
            # 生成随机数
            r1 = np.random.random(n_dims)
            r2 = np.random.random(n_dims)
            
            # 更新速度
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_best_positions[i] - positions[i]) +
                           c2 * r2 * (global_best_position - positions[i]))
            
            # 更新位置
            positions[i] += velocities[i]
            
            # 边界处理：反弹
            for d in range(n_dims):
                if positions[i, d] < bounds[d, 0]:
                    positions[i, d] = bounds[d, 0]
                    velocities[i, d] *= -0.5
                elif positions[i, d] > bounds[d, 1]:
                    positions[i, d] = bounds[d, 1]
                    velocities[i, d] *= -0.5
            
            # 计算适应度
            fitness = objective_func(positions[i])
            
            # 更新个体最优
            if (minimize and fitness < personal_best_fitness[i]) or \
               (not minimize and fitness > personal_best_fitness[i]):
                personal_best_positions[i] = positions[i].copy()
                personal_best_fitness[i] = fitness
                
                # 更新全局最优
                if (minimize and fitness < global_best_fitness) or \
                   (not minimize and fitness > global_best_fitness):
                    global_best_position = positions[i].copy()
                    global_best_fitness = fitness
                    print(f"> Iter {iteration}, New best: f({global_best_position}) = {global_best_fitness:.6f}")
        
        fitness_history.append(global_best_fitness)
    
    return global_best_position, global_best_fitness, fitness_history


def adaptive_pso(objective_func, bounds, n_particles, n_iter, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, minimize=True):
    """
    自适应粒子群算法，惯性权重随迭代线性递减
    """
    n_dims = len(bounds)
    bounds = np.array(bounds)
    
    # 初始化
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_particles, n_dims))
    velocities = np.random.uniform(-1, 1, (n_particles, n_dims))
    
    personal_best_positions = positions.copy()
    personal_best_fitness = np.array([objective_func(pos) for pos in positions])
    
    if minimize:
        global_best_idx = np.argmin(personal_best_fitness)
    else:
        global_best_idx = np.argmax(personal_best_fitness)
        
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_fitness = personal_best_fitness[global_best_idx]
    
    fitness_history = [global_best_fitness]
    
    for iteration in range(n_iter):
        # 线性递减惯性权重
        w = w_max - (w_max - w_min) * iteration / n_iter
        
        for i in range(n_particles):
            r1 = np.random.random(n_dims)
            r2 = np.random.random(n_dims)
            
            # 更新速度
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_best_positions[i] - positions[i]) +
                           c2 * r2 * (global_best_position - positions[i]))
            
            # 速度限制
            v_max = 0.2 * (bounds[:, 1] - bounds[:, 0])
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            # 更新位置
            positions[i] += velocities[i]
            
            # 边界处理
            positions[i] = np.clip(positions[i], bounds[:, 0], bounds[:, 1])
            
            # 计算适应度并更新
            fitness = objective_func(positions[i])
            
            if (minimize and fitness < personal_best_fitness[i]) or \
               (not minimize and fitness > personal_best_fitness[i]):
                personal_best_positions[i] = positions[i].copy()
                personal_best_fitness[i] = fitness
                
                if (minimize and fitness < global_best_fitness) or \
                   (not minimize and fitness > global_best_fitness):
                    global_best_position = positions[i].copy()
                    global_best_fitness = fitness
                    print(f"> Iter {iteration}, New best: f({global_best_position}) = {global_best_fitness:.6f}")
        
        fitness_history.append(global_best_fitness)
    
    return global_best_position, global_best_fitness, fitness_history


if __name__ == '__main__':
    # --- 使用示例1: 求解Sphere函数最小值 ---
    
    # 1. 定义目标函数：Sphere函数 f(x) = sum(x_i^2)
    def sphere_function(x):
        return np.sum(x**2)
    
    # 2. 设置参数
    bounds = [[-5, 5], [-5, 5]]  # 2维问题，每维范围[-5, 5]
    n_particles = 30             # 粒子数量
    n_iter = 100                 # 迭代次数
    
    # 3. 运行标准PSO
    print("=== 标准粒子群算法 ===")
    best_pos, best_fit, history = particle_swarm_optimization(
        sphere_function, bounds, n_particles, n_iter, minimize=True
    )
    
    print("\n--- 结果 ---")
    print(f"最优解: {best_pos}")
    print(f"最优值: {best_fit:.6f}")
    
    # 4. 运行自适应PSO
    print("\n=== 自适应粒子群算法 ===")
    best_pos_adaptive, best_fit_adaptive, history_adaptive = adaptive_pso(
        sphere_function, bounds, n_particles, n_iter, minimize=True
    )
    
    print("\n--- 结果 ---")
    print(f"最优解: {best_pos_adaptive}")
    print(f"最优值: {best_fit_adaptive:.6f}")
    
    # 5. 结果比较可视化
    plt.figure(figsize=(12, 5))
    
    # 收敛曲线对比
    plt.subplot(1, 2, 1)
    plt.plot(history, label='Standard PSO', marker='o', markersize=2)
    plt.plot(history_adaptive, label='Adaptive PSO', marker='s', markersize=2)
    plt.xlabel('迭代次数')
    plt.ylabel('最优适应度')
    plt.title('PSO收敛曲线对比')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 函数等高线和最优解
    plt.subplot(1, 2, 2)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.scatter(best_pos[0], best_pos[1], color='red', s=100, marker='*', label='Standard PSO')
    plt.scatter(best_pos_adaptive[0], best_pos_adaptive[1], color='blue', s=100, marker='*', label='Adaptive PSO')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Sphere函数等高线与最优解')
    plt.legend()
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # --- 使用示例2: 多峰函数Rastrigin ---
    print("\n=== 复杂多峰函数测试: Rastrigin函数 ===")
    
    def rastrigin_function(x):
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    bounds_rastrigin = [[-5.12, 5.12], [-5.12, 5.12]]
    
    best_pos_ras, best_fit_ras, history_ras = adaptive_pso(
        rastrigin_function, bounds_rastrigin, 50, 200, minimize=True
    )
    
    print(f"\nRastrigin函数最优解: {best_pos_ras}")
    print(f"Rastrigin函数最优值: {best_fit_ras:.6f}")
    print(f"理论最优值: 0.0")
    
    # 如何修改为你自己的问题:
    # 1. 修改 `objective_function` 为你的优化目标
    # 2. 修改 `bounds` 为你所有决策变量的取值范围
    # 3. 根据问题复杂度调整 `n_particles`, `n_iter` 等参数
    # 4. 选择 `minimize=True` (最小化) 或 `minimize=False` (最大化)