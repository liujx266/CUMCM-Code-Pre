# M_C4_AHP.py

import numpy as np

def ahp(matrix, method='eigenvalue'):
    """
    使用层次分析法(AHP)计算权重向量
    
    参数:
    matrix: ndarray, 判断矩阵 (n×n, 满足aij=1/aji, aii=1)
    method: str, 计算方法 'eigenvalue'(特征值法) 或 'geometric'(几何平均法)
    
    返回:
    weights: array, 权重向量
    CR: float, 一致性比率 (CR<0.1为可接受)
    """
    A = np.array(matrix, dtype=float)
    n = A.shape[0]
    
    # 随机一致性指标RI表
    RI = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49]
    
    if method == 'eigenvalue':
        # 特征值法计算权重
        eigenvalues, eigenvectors = np.linalg.eig(A)
        max_idx = np.argmax(np.real(eigenvalues))
        lambda_max = np.real(eigenvalues[max_idx])
        weights = np.abs(np.real(eigenvectors[:, max_idx]))
        weights = weights / np.sum(weights)
        
    elif method == 'geometric':
        # 几何平均法计算权重
        geometric_mean = np.power(np.prod(A, axis=1), 1/n)
        weights = geometric_mean / np.sum(geometric_mean)
        # 计算最大特征值
        Aw = np.dot(A, weights)
        lambda_max = np.mean(Aw / weights)
    
    # 一致性检验
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    CR = CI / RI[n-1] if n <= len(RI) and RI[n-1] > 0 else 0
    
    return weights, CR

if __name__ == '__main__':
    # --- 使用示例: 供应商选择问题的准则权重确定 ---
    # 准则: 价格、质量、服务、交期
    
    # 1. 构建判断矩阵 (专家两两比较结果)
    # 判断标度: 1(同等重要), 3(稍重要), 5(明显重要), 7(强烈重要), 9(极端重要)
    criteria_matrix = np.array([
        [1,   1/3, 1/2, 1/4],  # 价格 相对于 [价格,质量,服务,交期]
        [3,   1,   2,   1/2],  # 质量 相对于 [价格,质量,服务,交期]  
        [2,   1/2, 1,   1/3],  # 服务 相对于 [价格,质量,服务,交期]
        [4,   2,   3,   1  ]   # 交期 相对于 [价格,质量,服务,交期]
    ])
    
    # 2. 计算权重
    weights, CR = ahp(criteria_matrix)
    
    # 3. 显示结果
    criteria_names = ['价格', '质量', '服务', '交期']
    print("AHP权重计算结果:")
    print("-" * 30)
    for i, (name, weight) in enumerate(zip(criteria_names, weights)):
        print(f"{name}: {weight:.4f} ({weight*100:.2f}%)")
    
    print(f"\n一致性检验:")
    print(f"一致性比率 CR = {CR:.4f}")
    if CR < 0.1:
        print("✓ 通过一致性检验 (CR < 0.1)")
    else:
        print("✗ 未通过一致性检验 (CR ≥ 0.1)，建议重新调整判断矩阵")
    
    # 如何修改为你自己的问题:
    # 1. 构建你的判断矩阵 `criteria_matrix`，使用1-9标度进行两两比较
    # 2. 确保矩阵满足 aij = 1/aji 和 aii = 1 的性质
    # 3. 检查一致性比率CR，如果CR ≥ 0.1需要重新调整矩阵
    # 4. 获得的权重向量可用于TOPSIS、加权评分等后续分析