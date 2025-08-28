# M_A2_NumericalIntegration.py

import numpy as np

def simpson_integration(func, a, b, n=1000):
    """
    使用Simpson法计算定积分
    
    参数:
    func: function, 被积函数
    a: float, 积分下限
    b: float, 积分上限  
    n: int, 分割区间数 (必须为偶数)
    
    返回:
    result: float, 积分值
    """
    if n % 2 != 0:
        n += 1  # 确保n为偶数
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    
    # Simpson 1/3公式
    result = h/3 * (y[0] + 4*np.sum(y[1::2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    
    return result

def trapezoidal_integration(func, a, b, n=1000):
    """
    使用梯形法计算定积分
    
    参数:
    func: function, 被积函数
    a: float, 积分下限
    b: float, 积分上限
    n: int, 分割区间数
    
    返回:
    result: float, 积分值
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    
    # 梯形公式
    result = h/2 * (y[0] + 2*np.sum(y[1:-1]) + y[-1])
    
    return result

def double_integration(func, x_bounds, y_bounds, nx=100, ny=100):
    """
    计算二重积分
    
    参数:
    func: function, 被积函数 f(x,y)
    x_bounds: tuple, x的积分区间 (x_min, x_max)
    y_bounds: function or tuple, y的积分区间
              如果是tuple: (y_min, y_max) - 矩形区域
              如果是function: (y_min_func, y_max_func) - 一般区域
    nx, ny: int, x和y方向的分割数
    
    返回:
    result: float, 二重积分值
    """
    x_min, x_max = x_bounds
    x_vals = np.linspace(x_min, x_max, nx + 1)
    
    total = 0.0
    for i, x in enumerate(x_vals):
        # 确定当前x对应的y积分区间
        if callable(y_bounds[0]) and callable(y_bounds[1]):
            y_min, y_max = y_bounds[0](x), y_bounds[1](x)
        else:
            y_min, y_max = y_bounds
        
        # 对y方向积分
        def integrand_y(y):
            return func(x, y)
        
        y_integral = simpson_integration(integrand_y, y_min, y_max, ny)
        
        # Simpson权重
        if i == 0 or i == nx:
            weight = 1
        elif i % 2 == 1:
            weight = 4
        else:
            weight = 2
            
        total += weight * y_integral
    
    hx = (x_max - x_min) / nx
    result = hx/3 * total
    
    return result

if __name__ == '__main__':
    # --- 使用示例: 计算常见积分 ---
    
    # 示例1: 计算 ∫[0,π] sin(x) dx = 2
    print("示例1: ∫[0,π] sin(x) dx")
    result1 = simpson_integration(np.sin, 0, np.pi)
    print(f"Simpson法结果: {result1:.6f}")
    print(f"理论值: {2.0:.6f}")
    print(f"误差: {abs(result1 - 2.0):.6f}")
    
    # 示例2: 计算 ∫[0,1] x² dx = 1/3
    print("\n示例2: ∫[0,1] x² dx")
    def quadratic(x):
        return x**2
    
    result2 = simpson_integration(quadratic, 0, 1)
    theoretical2 = 1/3
    print(f"Simpson法结果: {result2:.6f}")
    print(f"理论值: {theoretical2:.6f}")
    print(f"误差: {abs(result2 - theoretical2):.6f}")
    
    # 示例3: 二重积分 ∫∫[0,1]×[0,1] (x²+y²) dxdy = 2/3
    print("\n示例3: ∫∫[0,1]×[0,1] (x²+y²) dxdy")
    def func_2d(x, y):
        return x**2 + y**2
    
    result3 = double_integration(func_2d, (0, 1), (0, 1), nx=50, ny=50)
    theoretical3 = 2/3
    print(f"二重积分结果: {result3:.6f}")
    print(f"理论值: {theoretical3:.6f}")
    print(f"误差: {abs(result3 - theoretical3):.6f}")
    
    # 如何修改为你自己的问题:
    # 1. 对于一元函数积分，定义你的函数，调用 simpson_integration(func, a, b)
    # 2. 对于二元函数积分，定义 f(x,y)，调用 double_integration(func, x_bounds, y_bounds)
    # 3. 增加分割数n可以提高精度，但会增加计算时间
    # 4. Simpson法比梯形法精度更高，适合光滑函数