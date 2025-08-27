# M_A1_RungeKutta.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def runge_kutta_4th(f, y0, t_span, h):
    """
    使用四阶龙格-库塔方法求解常微分方程(组) y' = f(t, y)

    参数:
    f: function, 微分方程的右端函数 f(t, y)
    y0: array, 初值条件
    t_span: tuple, 求解的起止时间 (t_start, t_end)
    h: float, 步长

    返回:
    t_points: array, 时间点
    y_points: array, 对应时间点的解，每一行是一个时间点的解
    """
    t_start, t_end = t_span
    t_points = np.arange(t_start, t_end + h, h)
    y_points = np.zeros((len(t_points), len(y0)))
    y_points[0] = y0
    
    y = y0.copy()
    for i in range(len(t_points) - 1):
        t = t_points[i]
        
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6.0
        y_points[i+1] = y
        
    return t_points, y_points

if __name__ == '__main__':
    # --- 使用示例: 求解洛伦兹吸引子系统 ---
    # y' = f(t, y) 其中 y = [x, y, z]
    # dx/dt = sigma * (y - x)
    # dy/dt = x * (rho - z) - y
    # dz/dt = x * y - beta * z
    
    # 1. 定义微分方程 f(t, y)
    def lorenz_system(t, y, sigma=10.0, rho=28.0, beta=8.0/3.0):
        x, y_val, z = y
        dxdt = sigma * (y_val - x)
        dydt = x * (rho - z) - y_val
        dzdt = x * y_val - beta * z
        return np.array([dxdt, dydt, dzdt])

    # 2. 设置初值、时间范围和步长
    y0 = np.array([0., 1., 20.])         # 初始状态 [x0, y0, z0]
    t_span = (0, 40)                     # 时间从0到40
    h = 0.01                             # 步长

    # 3. 调用求解器
    t_points, y_points = runge_kutta_4th(lambda t, y: lorenz_system(t, y), y0, t_span, h)

    # 4. 结果可视化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y_points[:, 0], y_points[:, 1], y_points[:, 2])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')  # type: ignore
    ax.set_title('Lorenz Attractor solved by 4th-order Runge-Kutta')
    plt.show()

    # 如何修改为你自己的问题:
    # 1. 修改 `lorenz_system` 函数为你自己的微分方程组。
    # 2. 修改 `y0`, `t_span`, `h` 为你问题的初始参数。
