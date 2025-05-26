import numpy as np
from scipy.integrate import quad

# 原函數
def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

# 基底函數 φ0 = 1, φ1 = x, φ2 = x^2
phi = [lambda x: 1,
       lambda x: x,
       lambda x: x**2]

# 內積定義：<f, g> = ∫ f(x)g(x) dx from -1 to 1
def inner_product(f1, f2):
    return quad(lambda x: f1(x) * f2(x), -1, 1)[0]

# 建立法方程組 A @ a = b
A = np.zeros((3, 3))
b = np.zeros(3)

for i in range(3):
    for j in range(3):
        A[i, j] = inner_product(phi[i], phi[j])
    b[i] = inner_product(f, phi[i])

# 解法方程組取得係數
a = np.linalg.solve(A, b)

# 定義逼近多項式 P2(x)
def P2(x):
    return a[0] + a[1] * x + a[2] * x**2

# 計算平方誤差
error = quad(lambda x: (f(x) - P2(x))**2, -1, 1)[0]

# 輸出結果
print("最小平方逼近 P2(x) = {:.6f} + {:.6f}x + {:.6f}x^2".format(a[0], a[1], a[2]))
print("平方誤差 (L2): {:.6f}".format(error))
