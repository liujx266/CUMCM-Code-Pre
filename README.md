# CUMCM算法库

> 全国大学生数学建模竞赛(CUMCM)核心算法实现库
> 
> 专为A、B、C三类题型设计的高效算法工具集

## 项目简介

本项目是针对全国大学生数学建模竞赛(CUMCM)精心设计的算法库，涵盖了A类工程建模、B类优化决策、C类统计分析三大类别的核心算法。每个算法都经过优化，提供完整的使用文档和实际应用示例。

### 核心特色

- **覆盖全面**: 包含14个核心算法，覆盖95%的CUMCM常用算法需求
- **即拿即用**: 每个算法都有清晰的接口设计和详细的使用示例
- **文档完善**: 提供算法原理、适用场景、参数说明和扩展应用
- **代码规范**: 统一的代码风格和注释规范，易于理解和修改
- **性能优化**: 基于成熟的科学计算库，保证算法效率和数值稳定性
- **智能推荐**: XGBoost等算法提供智能参数推荐功能

## 算法库结构

```
CUMCM-Code-Pre/
├── A/                     # A类 - 工程技术建模
│   ├── M_A1_RungeKutta.py           # 四阶龙格-库塔方法
│   ├── M_A2_NumericalIntegration.py # 数值积分方法
│   ├── A1.md                        # 算法说明文档
│   └── A2.md
├── B/                     # B类 - 优化决策算法  
│   ├── M_B1_GeneticAlgorithm.py          # 遗传算法
│   ├── M_B2_Dijkstra.py                  # 最短路径算法
│   ├── M_B3_ParticleSwarmOptimization.py # 粒子群优化
│   ├── M_B4_SimulatedAnnealing.py        # 模拟退火算法
│   ├── M_B5_LinearProgramming.py         # 线性规划算法
│   ├── B1.md ~ B5.md                     # 算法说明文档
│   └── ...
├── C/                     # C类 - 统计分析建模
│   ├── M_C1_PCA.py                  # 主成分分析
│   ├── M_C2_ARIMA.py               # 时间序列预测
│   ├── M_C3_TOPSIS.py              # 多属性决策分析
│   ├── M_C4_AHP.py                 # 层次分析法
│   ├── M_C5_MultipleRegression.py  # 多元回归分析
│   ├── M_C6_ClusterAnalysis.py     # 聚类分析
│   ├── M_C7_XGBoost.py             # XGBoost梯度提升
│   ├── C1.md ~ C7.md               # 算法说明文档
│   └── ...
└── README.md              # 项目说明文档
```

## 环境要求

### Python版本
- Python 3.7+

### 依赖库
```bash
pip install numpy scipy scikit-learn matplotlib pandas statsmodels xgboost
```

### 可选依赖
```bash
pip install seaborn plotly jupyter  # 用于高级可视化和交互式开发
```

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/your-username/CUMCM-Code-Pre.git
cd CUMCM-Code-Pre
```

### 2. 安装依赖
```bash
pip install -r requirements.txt  # 如果有requirements.txt文件
# 或者手动安装核心依赖
pip install numpy scipy scikit-learn matplotlib pandas statsmodels xgboost
```

### 3. 运行示例
```python
# A类示例：数值积分
from A.M_A2_NumericalIntegration import simpson_integration
import numpy as np

result = simpson_integration(np.sin, 0, np.pi, n=1000)
print(f"∫[0,π] sin(x) dx = {result:.6f}")  # 理论值: 2.0

# B类示例：模拟退火优化
from B.M_B4_SimulatedAnnealing import simulated_annealing

def rastrigin(x):
    return 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)

bounds = [[-5, 5], [-5, 5]]
best_sol, best_val, _ = simulated_annealing(rastrigin, bounds)
print(f"最优解: {best_sol}, 最优值: {best_val:.4f}")

# C类示例：层次分析法权重计算
from C.M_C4_AHP import ahp
import numpy as np

matrix = np.array([[1, 3, 1/2], [1/3, 1, 1/4], [2, 4, 1]])
weights, CR = ahp(matrix)
print(f"权重向量: {weights}")
print(f"一致性比率: {CR:.4f}")

# C类示例：XGBoost机器学习预测
from C.M_C7_XGBoost import xgb_classification, feature_importance_analysis

# 模拟数据
X = np.random.rand(1000, 5)
y = ['类别A' if x[0] + x[1] > 1 else '类别B' for x in X]

# 分类建模
model, results = xgb_classification(X, y)
print(f"测试准确率: {results['test_accuracy']:.4f}")

# 特征重要性
importance = feature_importance_analysis(model)
print(importance.head(3))
```

## 算法详览

### A类 - 工程技术建模 (2个算法)

| 算法 | 文件名 | 主要用途 | 典型应用 |
|------|--------|----------|----------|
| **四阶龙格-库塔法** | `M_A1_RungeKutta.py` | 常微分方程数值解 | 人口模型、传染病模型、动力学系统 |
| **数值积分** | `M_A2_NumericalIntegration.py` | 定积分数值计算 | 面积体积计算、物理量积分、概率计算 |

### B类 - 优化决策算法 (5个算法)

| 算法 | 文件名 | 主要用途 | 典型应用 |
|------|--------|----------|----------|
| **遗传算法** | `M_B1_GeneticAlgorithm.py` | 全局优化、参数搜索 | 函数优化、神经网络训练 |
| **最短路径** | `M_B2_Dijkstra.py` | 图论最短路径 | 交通规划、网络路由 |
| **粒子群优化** | `M_B3_ParticleSwarmOptimization.py` | 连续函数优化 | 参数估计、工程设计优化 |
| **模拟退火** | `M_B4_SimulatedAnnealing.py` | 组合优化、全局搜索 | TSP问题、调度问题 |
| **线性规划** | `M_B5_LinearProgramming.py` | 线性约束优化 | 资源分配、生产计划 |

### C类 - 统计分析建模 (7个算法)

| 算法 | 文件名 | 主要用途 | 典型应用 |
|------|--------|----------|----------|
| **主成分分析** | `M_C1_PCA.py` | 降维、特征提取 | 数据压缩、可视化 |
| **时间序列预测** | `M_C2_ARIMA.py` | 时序数据预测 | 销售预测、经济指标预测 |
| **多属性决策** | `M_C3_TOPSIS.py` | 方案评价排序 | 供应商选择、投资决策 |
| **层次分析法** | `M_C4_AHP.py` | 权重确定 | 指标权重、决策权重 |
| **多元回归** | `M_C5_MultipleRegression.py` | 预测建模 | 房价预测、影响因素分析 |
| **聚类分析** | `M_C6_ClusterAnalysis.py` | 数据分类 | 客户细分、市场分析 |
| **XGBoost** | `M_C7_XGBoost.py` | 集成学习预测 | 分类预测、特征重要性分析 |

## 使用指南

### 算法选择建议

**A题（工程技术类）推荐组合：**
- 龙格-库塔法 + 数值积分 + 优化算法
- 适用于：物理建模、工程设计、动态系统

**B题（优化决策类）推荐组合：**
- 遗传算法/模拟退火 + 线性规划 + 图论算法  
- 适用于：资源调度、路径规划、参数优化

**C题（统计分析类）推荐组合：**
- AHP权重 + XGBoost/回归预测 + 聚类分类 + TOPSIS决策
- 适用于：数据挖掘、趋势分析、综合评价、机器学习建模

### 代码使用模式

每个算法文件都遵循统一的使用模式：

```python
# 1. 导入算法
from [类别].M_[类别][编号]_[算法名] import [主函数名]

# 2. 准备数据
# 根据具体算法准备输入数据

# 3. 设置参数  
# 根据问题特点调整算法参数

# 4. 运行算法
result = algorithm_function(data, parameters)

# 5. 分析结果
# 每个文件的if __name__ == '__main__'部分都有详细示例
```

## 性能基准

### 算法复杂度对比

| 算法类别 | 时间复杂度 | 空间复杂度 | 适用数据规模 |
|----------|------------|------------|--------------|
| 数值积分 | O(n) | O(1) | 大规模 |
| 遗传算法 | O(g×p×n) | O(p×n) | 中等规模 |
| 线性规划 | O(n³) | O(n²) | 中大规模 |
| 聚类分析 | O(n²k) | O(nk) | 中等规模 |

*注：g=代数，p=种群大小，n=变量维数，k=聚类数*

## 竞赛应用实例

### 历年真题算法对应

**2024年题目：**
- A题："板凳龙"闹元宵 → 龙格-库塔法 + 数值积分
- B题：生产过程决策 → 模拟退火 + 线性规划  
- C题：农作物种植策略 → AHP + 多元回归 + 聚类

**2023年题目：**
- A题：定日镜场优化 → 粒子群优化 + 数值积分
- B题：多波束测线 → 遗传算法 + 图论算法
- C题：商品定价决策 → ARIMA + TOPSIS + AHP

## 开发指南

### 贡献代码

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/新算法名`)
3. 按照现有代码风格实现算法
4. 添加完整的文档和示例
5. 提交代码 (`git commit -am 'feat: 添加新算法'`)
6. 推送到分支 (`git push origin feature/新算法名`)
7. 创建Pull Request

### 代码规范

- **文件命名**: `M_[类别][编号]_[算法名].py`
- **函数命名**: 使用下划线分隔的小写字母
- **注释规范**: 每个函数都有详细的docstring
- **示例代码**: 每个文件都包含完整的使用示例
- **错误处理**: 适当的参数验证和异常处理

## 学习资源

### 推荐书籍
- 《数学建模算法与应用》- 司守奎
- 《Python科学计算基础教程》- 张若愚
- 《机器学习实战》- Peter Harrington

### 在线资源  
- [全国大学生数学建模竞赛官网](http://www.mcm.edu.cn/)
- [数学建模社区](https://www.shumo.com/)
- [算法可视化平台](https://algorithm-visualizer.org/)

## 贡献者

本项目由 [liujx266](https://github.com/liujx266) 维护

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

