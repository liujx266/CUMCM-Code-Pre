# M_C7_XGBoost.py

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

def xgb_classification(X, y, test_size=0.2, random_state=42, param_grid=None, cv_folds=5):
    """
    XGBoost分类分析
    
    参数:
    X: array, 特征矩阵 (样本×特征)
    y: array, 目标变量 (分类标签)
    test_size: float, 测试集比例
    random_state: int, 随机种子
    param_grid: dict, 超参数网格搜索范围
    cv_folds: int, 交叉验证折数
    
    返回:
    model: 训练好的XGBoost分类模型
    results: dict, 包含预测结果、评估指标、特征重要性等
    """
    # 标签编码 (XGBoost要求标签从0开始)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # 默认参数网格
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    
    # 创建XGBoost分类器
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob' if len(np.unique(y_encoded)) > 2 else 'binary:logistic',
        random_state=random_state,
        eval_metric='mlogloss' if len(np.unique(y_encoded)) > 2 else 'logloss'
    )
    
    # 网格搜索寻找最佳参数
    grid_search = GridSearchCV(
        xgb_clf, param_grid, cv=cv_folds, 
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # 预测
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    
    # 评估指标
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # 多分类和二分类的不同处理
    if len(np.unique(y_encoded)) > 2:
        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')
    else:
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
    
    # 特征重要性
    feature_importance = best_model.feature_importances_
    
    # 交叉验证得分
    cv_scores = cross_val_score(best_model, X, y_encoded, cv=cv_folds, scoring='accuracy')
    
    # 整理结果
    results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'y_train_pred': label_encoder.inverse_transform(y_train_pred),
        'y_test_pred': label_encoder.inverse_transform(y_test_pred),
        'y_test_proba': y_test_proba,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': label_encoder.inverse_transform(y_train),
        'y_test': label_encoder.inverse_transform(y_test),
        'label_encoder': label_encoder,
        'class_names': label_encoder.classes_
    }
    
    return best_model, results

def xgb_regression(X, y, test_size=0.2, random_state=42, param_grid=None, cv_folds=5):
    """
    XGBoost回归分析
    
    参数:
    X: array, 特征矩阵 (样本×特征)
    y: array, 目标变量 (连续值)
    test_size: float, 测试集比例
    random_state: int, 随机种子
    param_grid: dict, 超参数网格搜索范围
    cv_folds: int, 交叉验证折数
    
    返回:
    model: 训练好的XGBoost回归模型
    results: dict, 包含预测结果、评估指标、特征重要性等
    """
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 默认参数网格
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    
    # 创建XGBoost回归器
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=random_state,
        eval_metric='rmse'
    )
    
    # 网格搜索寻找最佳参数
    grid_search = GridSearchCV(
        xgb_reg, param_grid, cv=cv_folds,
        scoring='r2', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # 预测
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # 评估指标
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # 特征重要性
    feature_importance = best_model.feature_importances_
    
    # 交叉验证得分
    cv_scores = cross_val_score(best_model, X, y, cv=cv_folds, scoring='r2')
    
    # 整理结果
    results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    return best_model, results

def feature_importance_analysis(model, feature_names=None, top_n=10):
    """
    特征重要性分析
    
    参数:
    model: 训练好的XGBoost模型
    feature_names: list, 特征名称列表
    top_n: int, 显示前N个重要特征
    
    返回:
    importance_df: DataFrame, 特征重要性排序结果
    """
    importance_scores = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'特征{i+1}' for i in range(len(importance_scores))]
    
    # 创建重要性数据框
    total_importance = importance_scores.sum()
    if total_importance > 0:
        importance_percentage = importance_scores / total_importance * 100
    else:
        importance_percentage = np.zeros_like(importance_scores)
    
    importance_df = pd.DataFrame({
        '特征名称': feature_names,
        '重要性得分': importance_scores,
        '重要性百分比': importance_percentage
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('重要性得分', ascending=False)
    importance_df = importance_df.head(top_n).reset_index(drop=True)
    importance_df['排名'] = range(1, len(importance_df) + 1)
    
    # 重新排列列顺序
    importance_df = importance_df[['排名', '特征名称', '重要性得分', '重要性百分比']]
    
    return importance_df

def intelligent_param_recommendation(n_samples, n_features):
    """
    根据数据特征智能推荐XGBoost参数
    
    参数:
    n_samples: int, 样本数量
    n_features: int, 特征数量
    
    返回:
    recommended_params: dict, 推荐的参数组合
    """
    # 计算特征样本比
    feature_sample_ratio = n_features / n_samples
    
    # 基础参数：根据数据规模确定
    if n_samples < 1000:
        # 小数据集：保守参数，快速收敛
        base_params = {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9
        }
    elif n_samples < 10000:
        # 中等数据集：平衡参数
        base_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    else:
        # 大数据集：复杂参数，精细学习
        base_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    # 特征维度调整：根据特征样本比进行正则化调整
    if feature_sample_ratio > 0.1:
        # 高维数据：强正则化，防止过拟合
        base_params['learning_rate'] = min(base_params['learning_rate'] * 0.5, 0.05)
        base_params['max_depth'] = min(base_params['max_depth'], 4)
        base_params['subsample'] = 0.7
        base_params['colsample_bytree'] = 0.7
        base_params['reg_alpha'] = 1.0  # L1正则化
        base_params['reg_lambda'] = 1.0  # L2正则化
        
    elif feature_sample_ratio > 0.05:
        # 中等维度：适度正则化
        base_params['learning_rate'] = base_params['learning_rate'] * 0.8
        base_params['subsample'] = 0.8
        base_params['colsample_bytree'] = 0.8
        base_params['reg_alpha'] = 0.1
        base_params['reg_lambda'] = 0.1
        
    # 样本特征综合权衡：处理极端情况
    if n_samples < 500 and n_features > 50:
        # 小样本高维：极强正则化
        base_params.update({
            'n_estimators': 30,
            'max_depth': 2,
            'learning_rate': 0.05,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0
        })
    
    return base_params

def xgb_early_stopping_training(X, y, test_size=0.2, random_state=42, 
                               early_stopping_rounds=10, validation_fraction=0.2):
    """
    带早停机制的XGBoost训练
    
    参数:
    X: array, 特征矩阵
    y: array, 目标变量  
    test_size: float, 测试集比例
    random_state: int, 随机种子
    early_stopping_rounds: int, 早停轮数
    validation_fraction: float, 验证集比例
    
    返回:
    model: 训练好的模型
    results: dict, 训练结果
    """
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 从训练集中分出验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=validation_fraction, random_state=random_state
    )
    
    # 使用智能参数推荐
    recommended_params = intelligent_param_recommendation(X.shape[0], X.shape[1])
    
    # 判断任务类型
    is_classification = len(np.unique(y)) < len(y) * 0.05  # 简单启发式判断
    
    if is_classification:
        model = xgb.XGBClassifier(
            objective='multi:softprob' if len(np.unique(y)) > 2 else 'binary:logistic',
            random_state=random_state,
            eval_metric='mlogloss' if len(np.unique(y)) > 2 else 'logloss',
            **recommended_params  # 使用推荐的参数
        )
        eval_metric = 'mlogloss' if len(np.unique(y)) > 2 else 'logloss'
    else:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=random_state,
            eval_metric='rmse',
            **recommended_params  # 使用推荐的参数
        )
        eval_metric = 'rmse'
    
    # 训练模型 (带早停) - 新版XGBoost API
    model.set_params(early_stopping_rounds=early_stopping_rounds)
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 预测和评估
    y_test_pred = model.predict(X_test)
    
    # 获取最佳迭代轮数
    best_iteration = getattr(model, 'best_iteration', model.n_estimators)
    if best_iteration is None:
        best_iteration = model.n_estimators
    
    results = {
        'best_iteration': best_iteration,
        'stopped_early': best_iteration < model.n_estimators,
        'y_test_pred': y_test_pred,
        'X_test': X_test,
        'y_test': y_test,
        'feature_importance': model.feature_importances_
    }
    
    if is_classification:
        results['test_accuracy'] = accuracy_score(y_test, y_test_pred)
    else:
        results['test_r2'] = r2_score(y_test, y_test_pred)
        results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return model, results

if __name__ == '__main__':
    # --- 使用示例1: 客户流失预测 (分类问题) ---
    print("示例1: 客户流失预测 - XGBoost分类")
    
    # 模拟客户数据
    np.random.seed(42)
    n_samples = 1000
    
    # 特征: 月消费金额、使用时长、客服投诉次数、年龄、满意度评分
    monthly_spend = np.random.exponential(50, n_samples)
    usage_time = np.random.normal(12, 4, n_samples)  # 月使用小时数
    complaints = np.random.poisson(0.5, n_samples)   # 投诉次数
    age = np.random.normal(35, 12, n_samples)
    satisfaction = np.random.normal(3.5, 1.0, n_samples)  # 1-5评分
    
    X_clf = np.column_stack([monthly_spend, usage_time, complaints, age, satisfaction])
    feature_names_clf = ['月消费金额', '月使用时长', '投诉次数', '年龄', '满意度评分']
    
    # 流失概率 = f(低消费, 低使用时长, 多投诉, 低满意度)
    churn_prob = (
        0.8 - monthly_spend/100 + complaints*0.3 - usage_time*0.05 
        - satisfaction*0.2 + np.random.normal(0, 0.1, n_samples)
    )
    y_clf = (churn_prob > 0.5).astype(int)  # 二分类: 0=留存, 1=流失
    y_clf_labels = ['留存', '流失']
    y_clf_text = [y_clf_labels[i] for i in y_clf]
    
    print(f"数据规模: {X_clf.shape[0]}个客户, {X_clf.shape[1]}个特征")
    print(f"流失率: {(y_clf==1).sum()/len(y_clf)*100:.1f}%")
    
    # XGBoost分类分析
    model_clf, results_clf = xgb_classification(X_clf, y_clf_text, test_size=0.3)
    
    print(f"\n分类结果:")
    print(f"最佳参数: {results_clf['best_params']}")
    print(f"测试集准确率: {results_clf['test_accuracy']:.4f}")
    print(f"精确率: {results_clf['precision']:.4f}")
    print(f"召回率: {results_clf['recall']:.4f}")
    print(f"F1得分: {results_clf['f1_score']:.4f}")
    print(f"交叉验证得分: {results_clf['cv_mean']:.4f} ± {results_clf['cv_std']:.4f}")
    
    # 特征重要性分析
    importance_clf = feature_importance_analysis(model_clf, feature_names_clf)
    print(f"\n特征重要性排名:")
    for _, row in importance_clf.iterrows():
        print(f"  {row['排名']}. {row['特征名称']}: {row['重要性百分比']:.1f}%")
    
    # --- 使用示例2: 销售额预测 (回归问题) ---
    print(f"\n" + "="*50)
    print("示例2: 销售额预测 - XGBoost回归")
    
    # 模拟销售数据
    np.random.seed(42)
    n_samples = 800
    
    # 特征: 广告投入、促销力度、季节指数、竞争对手数量、历史销量
    ad_spend = np.random.exponential(10, n_samples)      # 广告投入(万元)
    promotion = np.random.uniform(0, 1, n_samples)       # 促销力度(0-1)
    season_index = np.random.normal(1, 0.3, n_samples)   # 季节指数
    competitors = np.random.poisson(3, n_samples)        # 竞争对手数量
    historical_sales = np.random.normal(50, 15, n_samples)  # 历史平均销量
    
    X_reg = np.column_stack([ad_spend, promotion, season_index, competitors, historical_sales])
    feature_names_reg = ['广告投入', '促销力度', '季节指数', '竞争对手数', '历史销量']
    
    # 销售额 = f(广告投入, 促销, 季节, 竞争, 历史销量) + 噪声
    y_reg = (
        ad_spend * 3 + promotion * 20 + season_index * 15 
        - competitors * 2 + historical_sales * 0.5 
        + np.random.normal(0, 5, n_samples)
    )
    
    print(f"数据规模: {X_reg.shape[0]}个样本, {X_reg.shape[1]}个特征")
    print(f"销售额范围: {y_reg.min():.1f} - {y_reg.max():.1f}")
    
    # XGBoost回归分析
    model_reg, results_reg = xgb_regression(X_reg, y_reg, test_size=0.3)
    
    print(f"\n回归结果:")
    print(f"最佳参数: {results_reg['best_params']}")
    print(f"测试集R²: {results_reg['test_r2']:.4f}")
    print(f"测试集RMSE: {results_reg['test_rmse']:.2f}")
    print(f"测试集MAE: {results_reg['test_mae']:.2f}")
    print(f"交叉验证R²: {results_reg['cv_mean']:.4f} ± {results_reg['cv_std']:.4f}")
    
    # 特征重要性分析
    importance_reg = feature_importance_analysis(model_reg, feature_names_reg)
    print(f"\n特征重要性排名:")
    for _, row in importance_reg.iterrows():
        print(f"  {row['排名']}. {row['特征名称']}: {row['重要性百分比']:.1f}%")
    
    # --- 使用示例3: 智能参数推荐 ---
    print(f"\n" + "="*50)
    print("示例3: 智能参数推荐")
    
    # 演示不同数据规模的参数推荐
    print(f"当前数据: {X_reg.shape[0]}样本, {X_reg.shape[1]}特征")
    recommended_params = intelligent_param_recommendation(X_reg.shape[0], X_reg.shape[1])
    print(f"推荐参数: {recommended_params}")
    
    # --- 使用示例4: 早停机制训练 ---
    print(f"\n" + "="*50)
    print("示例4: 早停机制训练 (使用智能推荐参数)")
    
    model_early, results_early = xgb_early_stopping_training(X_reg, y_reg, early_stopping_rounds=20)
    
    print(f"最佳迭代轮数: {results_early['best_iteration']}")
    print(f"是否早停: {'是' if results_early['stopped_early'] else '否'}")
    if 'test_r2' in results_early:
        print(f"测试集R²: {results_early['test_r2']:.4f}")
    
    # 如何修改为你自己的问题:
    # 1. 准备特征矩阵X和目标变量y
    # 2. 选择分类或回归问题类型
    # 3. 调整超参数网格搜索范围 (根据数据规模和计算资源)
    # 4. 分析特征重要性，理解模型决策依据
    # 5. 使用早停机制防止过拟合
    # 6. 根据业务需求选择合适的评估指标