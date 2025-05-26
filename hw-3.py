import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 設定輸出格式
pd.set_option('display.float_format', '{:.6f}'.format)

# (a) 計算 S₄(x)
m = 16
x = np.array([j / m for j in range(m)])
f_x = x**2 * np.sin(x)

N = 4
a = np.zeros(N + 1)
b = np.zeros(N + 1)

for k in range(N + 1):
    a[k] = (2 / m) * np.sum(f_x * np.cos(2 * np.pi * k * x))
for k in range(1, N + 1):
    b[k] = (2 / m) * np.sum(f_x * np.sin(2 * np.pi * k * x))

# 顯示係數
print("(a) S₄(x) 的係數：")
print(f"a0 = {a[0]:.6f}")
for k in range(1, N + 1):
    print(f"a{k} = {a[k]:.6f} , b{k} = {b[k]:.6f}")

# 定義 S₄(x)
def S4(x_val):
    result = a[0] / 2
    for k in range(1, N + 1):
        result += a[k] * np.cos(2 * np.pi * k * x_val)
        result += b[k] * np.sin(2 * np.pi * k * x_val)
    return result

# (b) ∫₀¹ S₄(x) dx
integral_S4 = quad(S4, 0, 1)[0]
print(f"\n(b) S₄(x) 積分 ≈ {integral_S4:.6f}")

# (c) 比較真實積分
real_integral = quad(lambda x: x**2 * np.sin(x), 0, 1)[0]
abs_error = abs(real_integral - integral_S4)
rel_error = abs_error / abs(real_integral) * 100

# 點誤差表格
x_points = np.linspace(0, 1, 2 * m)
f_exact = x_points**2 * np.sin(x_points)
s4_approx = np.array([S4(xi) for xi in x_points])
point_errors = np.abs(f_exact - s4_approx)

df = pd.DataFrame({
    'x': x_points,
    'f(x)': f_exact,
    'S₄(x)': s4_approx,
    'point_error': point_errors
})

print("\n(c) 各點比較：")
print(df)

# (d) 誤差顯示
square_error = np.sum((f_exact - s4_approx)**2)
print(f"\n(d) 精確積分 ≈ {real_integral:.6f}")
print(f"    絕對誤差 = {abs_error:.6e}")
print(f"    相對誤差 = {rel_error:.6f}%")
print(f"    平方誤差 = {square_error:.6f}")

# 畫圖
plt.plot(x_points, f_exact, label='f(x) = $x^2 \\sin x$', color='blue')
plt.plot(x_points, s4_approx, label='$S_4(x)$', color='red', linestyle='--')
plt.title("f(x) vs. S₄(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
