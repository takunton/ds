import numpy as np

# 損失関数計算
def const_func(x, y, theta_0, theta_1):
    # 残差を計算
    se = np.square(y - (theta_0 + theta_1 * x))

    # mseを計算
    result = np.mean(se)
    return result

# パラメータ更新関数
def update_theta(x, y, theta_0, theta_1, alpha):
    # 微分値計算
    theta_0_lean = 2 * np.mean((theta_0 + theta_1 * x) - y)
    theta_1_lean = 2 * np.mean(((theta_0 + theta_1 * x) - y) * x)
    
    # パラメータ更新
    new_theta_0 = theta_0 - alpha * theta_0_lean
    new_theta_1 = theta_1 - alpha * theta_1_lean
    
    return new_theta_0, new_theta_1