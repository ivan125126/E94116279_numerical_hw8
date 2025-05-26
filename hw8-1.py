import numpy as np
import matplotlib.pyplot as plt

# 資料點
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# -------- (a) 二次多項式擬合: y ≈ a + bx + cx^2 --------
A_poly = np.vstack([np.ones_like(x), x, x**2]).T
coeffs_poly = np.linalg.lstsq(A_poly, y, rcond=None)[0]
y_pred_poly = A_poly @ coeffs_poly
error_poly = np.sum((y - y_pred_poly)**2)

# -------- (b) 指數模型擬合: y ≈ b * e^(a*x) => ln(y) = ln(b) + a*x --------
Y_log = np.log(y)
A_exp = np.vstack([np.ones_like(x), x]).T
coeffs_exp = np.linalg.lstsq(A_exp, Y_log, rcond=None)[0]
a_exp, ln_b_exp = coeffs_exp[1], coeffs_exp[0]
b_exp = np.exp(ln_b_exp)
y_pred_exp = b_exp * np.exp(a_exp * x)
error_exp = np.sum((y - y_pred_exp)**2)

# -------- (c) 幂次模型擬合: y ≈ b * x^a => ln(y) = ln(b) + a*ln(x) --------
X_log = np.log(x)
Y_log = np.log(y)
A_pow = np.vstack([np.ones_like(x), X_log]).T
coeffs_pow = np.linalg.lstsq(A_pow, Y_log, rcond=None)[0]
a_pow, ln_b_pow = coeffs_pow[1], coeffs_pow[0]
b_pow = np.exp(ln_b_pow)
y_pred_pow = b_pow * x**a_pow
error_pow = np.sum((y - y_pred_pow)**2)

# -------- 輸出結果 --------
print("===== (a) 二次多項式擬合 =====")
print(f"模型: y ≈ {coeffs_poly[0]:.4f} + {coeffs_poly[1]:.4f}x + {coeffs_poly[2]:.4f}x^2")
print("平方總誤差:", error_poly)

print("\n===== (b) 指數模型擬合 =====")
print(f"模型: y ≈ {b_exp:.4f} * e^({a_exp:.4f}x)")
print("平方總誤差:", error_exp)

print("\n===== (c) 幂次模型擬合 =====")
print(f"模型: y ≈ {b_pow:.4f} * x^{a_pow:.4f}")
print("平方總誤差:", error_pow)

# -------- 畫圖比較 --------
x_plot = np.linspace(min(x), max(x), 200)
y_plot_poly = coeffs_poly[0] + coeffs_poly[1]*x_plot + coeffs_poly[2]*x_plot**2
y_plot_exp = b_exp * np.exp(a_exp * x_plot)
y_plot_pow = b_pow * x_plot**a_pow

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='原始資料')

plt.plot(x_plot, y_plot_poly, 'r-', label='(a) 二次多項式')
plt.plot(x_plot, y_plot_exp, 'g--', label='(b) 指數模型')
plt.plot(x_plot, y_plot_pow, 'b-.', label='(c) 幂次模型')

plt.xlabel("x")
plt.ylabel("y")
plt.title("最小二乘法擬合：多項式、指數、幂次")
plt.legend()
plt.grid(True)
plt.show()
