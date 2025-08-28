# M_C6_ClusterAnalysis.py

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def kmeans_clustering(X, n_clusters=3, random_state=42, standardize=True):
    """
    K-means聚类分析
    
    参数:
    X: array, 数据矩阵 (样本×特征)
    n_clusters: int, 聚类数量
    random_state: int, 随机种子
    standardize: bool, 是否标准化数据
    
    返回:
    labels: array, 聚类标签
    centers: array, 聚类中心
    results: dict, 聚类结果和评估指标
    """
    # 数据标准化
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        scaler = None
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # 获取聚类中心 (原始数据空间)
    if standardize:
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
    else:
        centers = kmeans.cluster_centers_
    
    # 评估指标
    silhouette = silhouette_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    inertia = kmeans.inertia_  # 类内平方和
    
    # 计算类内距离和类间距离
    within_cluster_distances = []
    for i in range(n_clusters):
        cluster_points = X_scaled[labels == i]
        if len(cluster_points) > 0:
            distances = np.sqrt(((cluster_points - kmeans.cluster_centers_[i])**2).sum(axis=1))
            within_cluster_distances.append(distances.mean())
        else:
            within_cluster_distances.append(0)
    
    results = {
        'labels': labels,
        'centers': centers,
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski_harabasz,
        'inertia': inertia,
        'within_cluster_distances': within_cluster_distances,
        'scaler': scaler,
        'model': kmeans
    }
    
    return labels, centers, results

def optimal_clusters(X, max_clusters=10, method='elbow'):
    """
    确定最优聚类数量
    
    参数:
    X: array, 数据矩阵
    max_clusters: int, 最大聚类数
    method: str, 选择方法 ('elbow', 'silhouette')
    
    返回:
    optimal_k: int, 最优聚类数
    scores: array, 各聚类数对应的得分
    """
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    k_range = range(2, max_clusters + 1)
    scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        if method == 'elbow':
            score = kmeans.inertia_  # 使用类内平方和 (越小越好)
        elif method == 'silhouette':
            score = silhouette_score(X_scaled, labels)  # 轮廓系数 (越大越好)
        
        scores.append(score)
    
    scores = np.array(scores)
    
    if method == 'elbow':
        # 肘部法: 寻找拐点
        # 简化方法: 计算二阶差分
        if len(scores) >= 3:
            second_diff = np.diff(np.diff(scores))
            optimal_k = k_range[np.argmax(second_diff) + 2]
        else:
            optimal_k = k_range[np.argmin(scores)]
    elif method == 'silhouette':
        # 轮廓系数法: 选择最大值
        optimal_k = k_range[np.argmax(scores)]
    
    return optimal_k, scores

def hierarchical_clustering(X, n_clusters=3, linkage='ward', standardize=True):
    """
    层次聚类分析
    
    参数:
    X: array, 数据矩阵
    n_clusters: int, 聚类数量
    linkage: str, 连接方法 ('ward', 'complete', 'average', 'single')
    standardize: bool, 是否标准化数据
    
    返回:
    labels: array, 聚类标签
    results: dict, 聚类结果
    """
    # 数据标准化
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        scaler = None
    
    # 层次聚类
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hierarchical.fit_predict(X_scaled)
    
    # 评估指标
    silhouette = silhouette_score(X_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
    
    results = {
        'labels': labels,
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski_harabasz,
        'scaler': scaler,
        'model': hierarchical
    }
    
    return labels, results

def dbscan_clustering(X, eps=0.5, min_samples=5, standardize=True):
    """
    DBSCAN密度聚类
    
    参数:
    X: array, 数据矩阵
    eps: float, 邻域半径
    min_samples: int, 最小样本数
    standardize: bool, 是否标准化数据
    
    返回:
    labels: array, 聚类标签 (-1表示噪声点)
    results: dict, 聚类结果
    """
    # 数据标准化
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        scaler = None
    
    # DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    # 统计信息
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    results = {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels),
        'scaler': scaler,
        'model': dbscan
    }
    
    # 如果有足够的聚类，计算轮廓系数
    if n_clusters > 1 and n_noise < len(labels):
        # 排除噪声点计算轮廓系数
        non_noise_indices = labels != -1
        if np.sum(non_noise_indices) > 0:
            silhouette = silhouette_score(X_scaled[non_noise_indices], 
                                        labels[non_noise_indices])
            results['silhouette_score'] = silhouette
    
    return labels, results

if __name__ == '__main__':
    # --- 使用示例: 客户细分问题 ---
    print("示例: 客户细分聚类分析")
    
    # 模拟客户数据
    np.random.seed(42)
    n_customers = 300
    
    # 生成三类客户数据
    # 类别1: 高消费，高频次
    cluster1 = np.random.multivariate_normal([80, 15], [[100, 20], [20, 25]], 100)
    # 类别2: 中等消费，中等频次  
    cluster2 = np.random.multivariate_normal([50, 8], [[60, 10], [10, 16]], 100)
    # 类别3: 低消费，低频次
    cluster3 = np.random.multivariate_normal([25, 4], [[40, 5], [5, 9]], 100)
    
    X = np.vstack([cluster1, cluster2, cluster3])
    feature_names = ['月消费金额', '月购买频次']
    
    print(f"数据规模: {X.shape[0]}个客户, {X.shape[1]}个特征")
    print(f"特征: {feature_names}")
    
    # 1. 确定最优聚类数
    print("\n1. 确定最优聚类数:")
    optimal_k_elbow, elbow_scores = optimal_clusters(X, max_clusters=8, method='elbow')
    optimal_k_silhouette, silhouette_scores = optimal_clusters(X, max_clusters=8, method='silhouette')
    
    print(f"肘部法推荐聚类数: {optimal_k_elbow}")
    print(f"轮廓系数法推荐聚类数: {optimal_k_silhouette}")
    
    # 2. K-means聚类
    print(f"\n2. K-means聚类 (k={optimal_k_silhouette}):")
    labels_kmeans, centers_kmeans, results_kmeans = kmeans_clustering(X, n_clusters=optimal_k_silhouette)
    
    print(f"轮廓系数: {results_kmeans['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz指数: {results_kmeans['calinski_harabasz_score']:.2f}")
    
    print("聚类中心:")
    for i, center in enumerate(centers_kmeans):
        print(f"  群体{i+1}: 月消费={center[0]:.1f}元, 购买频次={center[1]:.1f}次")
    
    # 统计各聚类的大小
    unique_labels, counts = np.unique(labels_kmeans, return_counts=True)
    print("聚类大小:")
    for label, count in zip(unique_labels, counts):
        print(f"  群体{label+1}: {count}个客户")
    
    # 3. 层次聚类对比
    print(f"\n3. 层次聚类对比:")
    labels_hierarchical, results_hierarchical = hierarchical_clustering(X, n_clusters=optimal_k_silhouette)
    print(f"轮廓系数: {results_hierarchical['silhouette_score']:.4f}")
    
    # 4. DBSCAN聚类
    print(f"\n4. DBSCAN密度聚类:")
    labels_dbscan, results_dbscan = dbscan_clustering(X, eps=1.0, min_samples=10)
    print(f"发现聚类数: {results_dbscan['n_clusters']}")
    print(f"噪声点数量: {results_dbscan['n_noise']} ({results_dbscan['noise_ratio']*100:.1f}%)")
    if 'silhouette_score' in results_dbscan:
        print(f"轮廓系数: {results_dbscan['silhouette_score']:.4f}")
    
    # 如何修改为你自己的问题:
    # 1. 准备数据矩阵X，每行是一个样本，每列是一个特征
    # 2. 使用optimal_clusters()确定最佳聚类数
    # 3. 选择合适的聚类算法 (K-means/层次聚类/DBSCAN)
    # 4. 评估聚类质量 (轮廓系数、Calinski-Harabasz指数等)
    # 5. 分析和解释聚类结果的实际意义