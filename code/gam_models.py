import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Step 1: 加载和预处理数据 ---

# 设置Matplotlib以支持中文显示，并指定字体和大小
try:
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置字体时出错: {e}")
    print("将使用Matplotlib默认字体。如果中文显示为方块，请确保'Microsoft YaHei'字体已安装。")

# 文件路径
file_path = '合并结果去重-手动去掉异常值并补零.xlsx'

# 读取Excel文件
try:
    df = pd.read_excel(file_path)
    # 重命名列以便于代码中使用（请根据你的实际列名修改）
    df.rename(columns={
        '检测孕周': 'Gestational_Week',
        '孕妇BMI': 'BMI',
        'Y染色体浓度': 'Y_conc'
    }, inplace=True)
    print("数据加载成功，数据前5行：")
    print(df.head())
except FileNotFoundError:
    print(f"错误：无法在指定路径找到文件 '{file_path}'。请确保文件存在。")
    exit()

# 数据清洗与转换
# 确保目标列是数值类型
df['Gestational_Week'] = pd.to_numeric(df['Gestational_Week'], errors='coerce')
df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
df['Y_conc'] = pd.to_numeric(df['Y_conc'], errors='coerce')

# 删除包含缺失值的行
df.dropna(subset=['Gestational_Week', 'BMI', 'Y_conc'], inplace=True)

# 筛选男胎数据 (假设Y_conc > 0 为男胎)
df_male = df[df['Y_conc'].notna() & (df['Y_conc'] > 0)].copy()

print(f"\n预处理后，用于建模的男胎样本数为: {len(df_male)}")

# --- Step 2: GAM模型构建与拟合 ---

# 准备建模数据
# 对Y染色体浓度进行asinsqrt变换
# 假设Y_conc已经是0-1之间的小数
df_male['y_transformed'] = np.arcsin(np.sqrt(df_male['Y_conc']))

# 选择特征和目标变量
X = df_male[['Gestational_Week', 'BMI']].values
y = df_male['y_transformed'].values

# 定义并拟合GAM模型
print("\n正在拟合GAM模型...")
gam = LinearGAM(s(0, n_splines=15) + s(1, n_splines=15)).fit(X, y)
print("模型拟合完成。")


# --- Step 3: 模型结果解读 ---

# 显示模型的统计摘要
print("\n--- GAM模型统计摘要 ---")
# 注意：我们将遵循pygam的警告，主要关注EDoF值，对p值只做定性参考。
# 增加一个注释来提醒自己和读者这一点。
print("注意: 以下P值可能偏小，请重点关注EDoF值来判断非线性。")
gam.summary()


# --- Step 4: 可视化部分效应图 ---

print("\n正在生成部分效应图...")

# 创建一个1x2的子图布局
fig, axes = plt.subplots(1, 2, figsize=(10, 4)) 

# ==================== 代码改动部分开始 ====================
# 绘制第一个特征（孕周）的部分效应图
term_idx_week = 0
XX_week = gam.generate_X_grid(term=term_idx_week)
pdep_week, confi_week = gam.partial_dependence(term=term_idx_week, X=XX_week, width=0.95)

# confi_week 是一个 (100, 2) 的数组，confi_week[:, 0] 是下界, confi_week[:, 1] 是上界
axes[0].plot(XX_week[:, term_idx_week], pdep_week, color='b', linewidth=1.5, label='平滑样条效应')
axes[0].fill_between(XX_week[:, term_idx_week], confi_week[:, 0], confi_week[:, 1], color='b', alpha=0.2, label='95% 置信带')
axes[0].set_title('孕周的部分效应 (Partial Effect of Gestational Week)')
axes[0].set_xlabel('检测孕周')
axes[0].set_ylabel('asinsqrt(Y浓度)的效应值')
axes[0].legend()


# 绘制第二个特征（BMI）的部分效应图
term_idx_bmi = 1
XX_bmi = gam.generate_X_grid(term=term_idx_bmi)
pdep_bmi, confi_bmi = gam.partial_dependence(term=term_idx_bmi, X=XX_bmi, width=0.95)

# 同样地，处理 confi_bmi
axes[1].plot(XX_bmi[:, term_idx_bmi], pdep_bmi, color='b', linewidth=1.5, label='平滑样条效应')
axes[1].fill_between(XX_bmi[:, term_idx_bmi], confi_bmi[:, 0], confi_bmi[:, 1], color='b', alpha=0.2, label='95% 置信带')
axes[1].set_title('孕妇BMI的部分效应 (Partial Effect of BMI)')
axes[1].set_xlabel('孕妇BMI')
axes[1].tick_params(axis='y', labelleft=False) # 隐藏Y轴标签
axes[1].legend()
# ==================== 代码改动部分结束 ====================


# 调整布局并保存图像
plt.tight_layout(pad=1.5)
plt.savefig('GAM_partial_effects_msyh_fixed.png', dpi=300)
print("\n部分效应图已保存为 'GAM_partial_effects_msyh_fixed.png'")
plt.show()

