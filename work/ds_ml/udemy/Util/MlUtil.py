import numpy as np

def const_func(x, y, theta_0, theta_1):
    # 残差を計算
    se = np.square(y - (theta_0 + theta_1 * x))

    # mseを計算
    result = np.mean(se)
    return result