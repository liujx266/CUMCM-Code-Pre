# M_B1_GeneticAlgorithm.py
import numpy as np
import matplotlib.pyplot as plt

def genetic_algorithm(objective_func, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    """
    使用遗传算法求解函数最大值问题

    参数:
    objective_func: function, 目标函数 f(x)
    bounds: array, 变量的取值范围, e.g., [[min1, max1], [min2, max2], ...]
    n_bits: int, 每个变量用多少位二进制编码
    n_iter: int, 迭代次数
    n_pop: int, 种群大小
    r_cross: float, 交叉概率
    r_mut: float, 变异概率
    
    返回:
    best_solution: array, 找到的最优解
    best_fitness: float, 最优解对应的函数值
    """
    # 将十进制数值解码为二进制字符串
    def decode(bounds, n_bits, bitstring):
        decoded = list()
        largest = 2**n_bits
        start = 0
        for i in range(len(bounds)):
            end = start + n_bits
            substring = bitstring[start:end]
            chars = ''.join([str(s) for s in substring])
            integer = int(chars, 2)
            value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
            decoded.append(value)
            start = end
        return decoded

    # 选择操作 (轮盘赌)
    def selection(pop, scores):
        total_score = sum(scores)
        if total_score == 0:  # 避免所有分数为0的情况
             selection_probs = [1.0/len(scores)] * len(scores)
        else:
             selection_probs = [s/total_score for s in scores]
        selected_indices = np.random.choice(len(pop), size=len(pop), p=selection_probs)
        return [pop[i] for i in selected_indices]
        
    # 交叉操作 (单点交叉)
    def crossover(p1, p2, r_cross):
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < r_cross:
            pt = np.random.randint(1, len(p1)-1)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return c1, c2

    # 变异操作
    def mutation(bitstring, r_mut):
        for i in range(len(bitstring)):
            if np.random.rand() < r_mut:
                bitstring[i] = 1 - bitstring[i]

    # 初始化种群
    pop = [np.random.randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
    best, best_eval = 0, -float('inf')

    for gen in range(n_iter):
        # 解码并计算适应度
        decoded = [decode(bounds, n_bits, p) for p in pop]
        scores = [objective_func(d) for d in decoded]
        
        # 寻找当前代的最优解
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(f"> Gen {gen}, New best: f({decode(bounds, n_bits, pop[i])}) = {scores[i]:.4f}")

        # 选择
        selected = selection(pop, scores)
        
        # 创建下一代
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            c1, c2 = crossover(p1, p2, r_cross)
            mutation(c1, r_mut)
            mutation(c2, r_mut)
            children.append(c1)
            children.append(c2)
        pop = children

    best_solution_decoded = decode(bounds, n_bits, best)
    return best_solution_decoded, best_eval

if __name__ == '__main__':
    # --- 使用示例: 求解函数 f(x) = x * sin(10*pi*x) + 2.0 在 [-1, 2] 上的最大值 ---
    
    # 1. 定义目标函数
    def objective_function(x):
        return x[0] * np.sin(10 * np.pi * x[0]) + 2.0

    # 2. 设置参数
    bounds = [[-1.0, 2.0]]       # 变量x的范围
    n_bits = 16                  # 编码位数
    n_iter = 100                 # 迭代次数
    n_pop = 100                  # 种群大小
    r_cross = 0.9                # 交叉概率
    r_mut = 1.0 / (float(n_bits) * len(bounds)) # 变异概率

    # 3. 运行遗传算法
    best_sol, best_val = genetic_algorithm(objective_function, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
    print("\n--- Done! ---")
    print(f"Best Solution x = {best_sol[0]:.4f}")
    print(f"Maximum Value f(x) = {best_val:.4f}")

    # 4. 可视化
    x_range = np.arange(bounds[0][0], bounds[0][1], 0.01)
    y_range = [objective_function([x]) for x in x_range]
    plt.plot(x_range, y_range)
    plt.scatter(best_sol, best_val, color='red', label=f'Maximum ({best_sol[0]:.2f}, {best_val:.2f})')
    plt.title("Genetic Algorithm Optimization")
    plt.legend()
    plt.show()

    # 如何修改为你自己的问题:
    # 1. 修改 `objective_function` 为你的优化目标，注意这里是求最大值。若求最小值，可在返回值前加负号。
    # 2. 修改 `bounds` 为你所有决策变量的取值范围。
    # 3. 调整 `n_bits`, `n_iter`, `n_pop`, `r_cross`, `r_mut` 等超参数以获得更好结果。
