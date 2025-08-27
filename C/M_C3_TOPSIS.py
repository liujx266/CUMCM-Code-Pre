# M_C3_TOPSIS.py

import numpy as np

def topsis(matrix, weights, impacts):
    """
    使用TOPSIS方法进行多属性决策分析

    参数:
    matrix: ndarray, 决策矩阵 (m个方案, n个指标)
    weights: array, 指标权重 (长度为n)
    impacts: list, 指标类型, '+'表示效益型, '-'表示成本型
    
    返回:
    scores: array, 各方案的最终得分
    ranking: array, 方案的排名 (从1开始)
    """
    # 1. 标准化矩阵
    norm_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0))
    
    # 2. 加权标准化矩阵
    weighted_matrix = norm_matrix * weights
    
    # 3. 确定正理想解和负理想解
    positive_ideal = np.zeros(weighted_matrix.shape[1])
    negative_ideal = np.zeros(weighted_matrix.shape[1])
    
    for i in range(weighted_matrix.shape[1]):
        if impacts[i] == '+':
            positive_ideal[i] = np.max(weighted_matrix[:, i])
            negative_ideal[i] = np.min(weighted_matrix[:, i])
        else: # impacts[i] == '-'
            positive_ideal[i] = np.min(weighted_matrix[:, i])
            negative_ideal[i] = np.max(weighted_matrix[:, i])
            
    # 4. 计算各方案到正负理想解的距离
    dist_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal)**2, axis=1))
    dist_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal)**2, axis=1))
    
    # 5. 计算最终得分
    scores = dist_negative / (dist_positive + dist_negative)
    
    # 6. 排名
    ranking = len(scores) - scores.argsort().argsort()
    
    return scores, ranking

if __name__ == '__main__':
    # --- 使用示例: 评价4个供应商的综合表现 ---
    # 指标: 价格(成本-), 质量(效益+), 交货时间(成本-), 服务(效益+)
    
    # 1. 构建决策矩阵 (4个供应商, 4个指标)
    decision_matrix = np.array([
        [250, 90, 10, 8],   # 供应商 A
        [280, 80, 12, 7],   # 供应商 B
        [220, 95, 8,  9],   # 供应商 C
        [300, 85, 15, 6]    # 供应商 D
    ])
    
    # 2. 设置权重和指标类型
    weights = np.array([0.3, 0.3, 0.2, 0.2])  # 权重总和应为1
    impacts = ['-', '+', '-', '+']  # 价格和交货时间为成本型，质量和服务为效益型
    
    # 3. 运行TOPSIS分析
    scores, ranking = topsis(decision_matrix, weights, impacts)
    
    # 4. 显示结果
    suppliers = ['供应商A', '供应商B', '供应商C', '供应商D']
    print("TOPSIS分析结果:")
    print("-" * 40)
    for i, supplier in enumerate(suppliers):
        print(f"{supplier}: 得分 = {scores[i]:.4f}, 排名 = {ranking[i]}")
        
    # 按得分排序显示
    print("\n按排名排序:")
    sorted_indices = np.argsort(-scores)  # 按得分降序排列
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"第{rank}名: {suppliers[idx]} (得分: {scores[idx]:.4f})")
    
    # 如何修改为你自己的问题:
    # 1. 构建你的决策矩阵 `decision_matrix`，行为方案，列为指标。
    # 2. 设置 `weights` 数组，确保权重总和为1。
    # 3. 设置 `impacts` 列表，'+'表示越大越好的效益型指标，'-'表示越小越好的成本型指标。
