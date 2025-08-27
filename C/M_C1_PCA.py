# M_C1_PCA.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # --- 使用示例: 对一个虚拟数据集进行PCA降维 ---
    # 1. 生成虚拟数据 (模拟玻璃成分数据)
    # 假设有100个样本，10个化学成分指标
    np.random.seed(42)
    X = np.random.rand(100, 10)
    # 让人为地创建一些相关性
    X[:, 1] = X[:, 0] * 2 + np.random.normal(0, 0.1, 100)
    X[:, 3] = -X[:, 2] + np.random.normal(0, 0.2, 100)
    
    # 2. 数据标准化 (非常重要的一步)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 应用PCA
    # n_components可以是整数，也可以是(0,1)之间的小数，表示保留的主成分方差贡献率阈值
    pca = PCA(n_components=2) # 降到2维
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. 分析结果
    print("PCA降维后的数据形状:", X_pca.shape)
    print("\n各主成分的方差贡献率:", pca.explained_variance_ratio_)
    print(f"前两个主成分累计方差贡献率: {sum(pca.explained_variance_ratio_):.2%}")
    
    # 5. 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Mock Dataset')
    plt.grid(True)
    plt.show()

    # 如何修改为你自己的问题:
    # 1. 将你的数据加载到 `X` 中，确保它是一个 NumPy 数组，形状为 (样本数, 特征数)。
    # 2. 标准化是必须的，直接使用 `StandardScaler` 即可。
    # 3. 决定 `n_components` 的值。可以先设置为一个较大的数(如0.95)，
    #    让PCA自动选择能解释95%方差的主成分数量，也可以直接指定降到的维度。
