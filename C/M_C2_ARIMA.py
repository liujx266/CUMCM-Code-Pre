# M_C2_ARIMA.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

if __name__ == '__main__':
    # --- 使用示例: 对一个虚拟的时间序列数据进行预测 ---
    # 1. 生成虚拟时间序列数据 (带趋势和季节性)
    np.random.seed(0)
    n_points = 100
    t = np.arange(n_points)
    trend = 0.5 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(0, 5, n_points)
    data = trend + seasonality + noise
    
    date_range = pd.date_range(start="2023-01-01", periods=n_points, freq='D')
    series = pd.Series(data, index=date_range)
    
    # 2. 创建并拟合ARIMA模型
    # order=(p, d, q) 是模型的参数
    # p: 自回归项数, d: 差分阶数, q: 移动平均项数
    # 这些参数通常需要通过ACF和PACF图来确定，这里为简化，直接指定
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    
    # 3. 打印模型摘要
    print(model_fit.summary())
    
    # 4. 进行预测
    forecast_steps = 20
    forecast = model_fit.get_forecast(steps=forecast_steps)
    # 计算预测开始日期（原序列最后一天的下一天）
    forecast_start = "2023-04-11"  # 2023-01-01 + 100天后的下一天
    forecast_index = pd.date_range(start=forecast_start, periods=forecast_steps, freq='D')
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
    confidence_intervals = forecast.conf_int()
    
    # 5. 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original Data')
    plt.plot(forecast_series, label='Forecast', color='red')
    plt.fill_between(forecast_index, 
                     confidence_intervals.iloc[:, 0], 
                     confidence_intervals.iloc[:, 1], 
                     color='pink', alpha=0.5, label='95% Confidence Interval')
    plt.title('ARIMA Forecast')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 如何修改为你自己的问题:
    # 1. 将你的时间序列数据加载成 Pandas Series，并确保索引是时间格式。
    # 2. 最重要的步骤是确定 `order=(p, d, q)`。
    #    - d: 对数据进行差分，直到数据平稳（没有明显趋势）。
    #    - p/q: 观察平稳后序列的ACF图和PACF图来确定。
    # 3. 使用 `model_fit.get_forecast()` 进行预测。
