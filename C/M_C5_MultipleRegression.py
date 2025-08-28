# M_C5_MultipleRegression.py

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def multiple_regression(X, y, method='ols', alpha=1.0, test_size=0.2, random_state=42):
    """
    多元回归分析
    
    参数:
    X: array, 特征矩阵 (样本×特征)
    y: array, 目标变量
    method: str, 回归方法 ('ols', 'ridge', 'lasso')
    alpha: float, 正则化参数 (仅用于ridge和lasso)
    test_size: float, 测试集比例
    random_state: int, 随机种子
    
    返回:
    model: 训练好的回归模型
    results: dict, 包含系数、评估指标等结果
    """
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 选择回归方法
    if method == 'ols':
        model = LinearRegression()
    elif method == 'ridge':
        model = Ridge(alpha=alpha)
    elif method == 'lasso':
        model = Lasso(alpha=alpha)
    else:
        raise ValueError("方法必须是 'ols', 'ridge' 或 'lasso'")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 评估指标
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # 整理结果
    results = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    return model, results

def polynomial_regression(X, y, degree=2, test_size=0.2, random_state=42):
    """
    多项式回归分析
    
    参数:
    X: array, 特征矩阵
    y: array, 目标变量
    degree: int, 多项式次数
    test_size: float, 测试集比例
    random_state: int, 随机种子
    
    返回:
    model: 训练好的回归模型
    poly_features: 多项式特征转换器
    results: dict, 回归结果
    """
    # 生成多项式特征
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    # 标准化特征 (多项式特征可能很大)
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    
    # 多元线性回归
    model, results = multiple_regression(X_poly_scaled, y, method='ols', 
                                       test_size=test_size, random_state=random_state)
    
    # 添加预处理器到结果中
    results['poly_features'] = poly_features
    results['scaler'] = scaler
    
    return model, poly_features, results

def feature_selection_regression(X, y, feature_names=None, alpha_range=None):
    """
    使用Lasso进行特征选择的回归分析
    
    参数:
    X: array, 特征矩阵
    y: array, 目标变量  
    feature_names: list, 特征名称
    alpha_range: array, 正则化参数范围
    
    返回:
    best_model: 最佳Lasso模型
    selected_features: 选中的特征索引
    results: dict, 分析结果
    """
    if alpha_range is None:
        alpha_range = np.logspace(-4, 1, 50)  # 0.0001 到 10
    
    if feature_names is None:
        feature_names = [f'特征{i+1}' for i in range(X.shape[1])]
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    best_alpha = None
    best_score = -np.inf
    best_model = None
    
    # 交叉验证选择最佳alpha
    from sklearn.model_selection import cross_val_score
    
    for alpha in alpha_range:
        model = Lasso(alpha=alpha, max_iter=2000)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        mean_score = scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
            best_model = model
    
    # 用最佳参数训练模型
    best_model.fit(X_scaled, y)
    
    # 找出选中的特征 (系数不为0)
    selected_indices = np.where(np.abs(best_model.coef_) > 1e-6)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    
    results = {
        'best_alpha': best_alpha,
        'best_cv_score': best_score,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'coefficients': best_model.coef_[selected_indices],
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    return best_model, selected_features, results

if __name__ == '__main__':
    # --- 使用示例: 房价预测问题 ---
    print("示例: 房价预测多元回归分析")
    
    # 模拟房价数据
    np.random.seed(42)
    n_samples = 200
    
    # 特征: 面积、房间数、楼层、装修年份、距离市中心距离
    area = np.random.normal(100, 30, n_samples)  # 面积(m²)
    rooms = np.random.poisson(3, n_samples)      # 房间数
    floor = np.random.randint(1, 21, n_samples)  # 楼层  
    age = np.random.randint(0, 31, n_samples)    # 房龄
    distance = np.random.exponential(5, n_samples)  # 距市中心距离(km)
    
    X = np.column_stack([area, rooms, floor, age, distance])
    feature_names = ['面积', '房间数', '楼层', '房龄', '距离']
    
    # 真实关系: 面积正相关，房间数正相关，楼层有最优值，房龄负相关，距离负相关
    y = (area * 0.05 + rooms * 2 + (floor - 10)**2 * (-0.01) - age * 0.1 
         - distance * 0.3 + np.random.normal(0, 1, n_samples) + 50)
    
    print(f"数据规模: {X.shape[0]}个样本, {X.shape[1]}个特征")
    
    # 1. 普通多元线性回归
    print("\n1. 多元线性回归结果:")
    model_ols, results_ols = multiple_regression(X, y, method='ols')
    
    print(f"训练集R²: {results_ols['train_r2']:.4f}")
    print(f"测试集R²: {results_ols['test_r2']:.4f}")
    print("回归系数:")
    for i, (name, coef) in enumerate(zip(feature_names, results_ols['coefficients'])):
        print(f"  {name}: {coef:.4f}")
    print(f"截距: {results_ols['intercept']:.4f}")
    
    # 2. Ridge回归 (处理多重共线性)
    print("\n2. Ridge回归结果:")
    model_ridge, results_ridge = multiple_regression(X, y, method='ridge', alpha=1.0)
    print(f"测试集R²: {results_ridge['test_r2']:.4f}")
    
    # 3. Lasso回归和特征选择
    print("\n3. Lasso特征选择结果:")
    model_lasso, selected_features, results_lasso = feature_selection_regression(
        X, y, feature_names
    )
    
    print(f"最佳正则化参数: {results_lasso['best_alpha']:.4f}")
    print(f"交叉验证得分: {results_lasso['best_cv_score']:.4f}")
    print(f"选中的特征: {selected_features}")
    
    # 如何修改为你自己的问题:
    # 1. 准备特征矩阵X和目标变量y
    # 2. 选择合适的回归方法 (ols/ridge/lasso)
    # 3. 评估模型性能 (R²、MSE等指标)
    # 4. 解释回归系数的实际意义
    # 5. 使用模型进行预测