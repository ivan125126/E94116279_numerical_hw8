import numpy as np

# 設定 16 個等距點
m = 16
x = np.array([j / m for j in range(m)])
f_x = x**2 * np.sin(x)

# 設定最高次數 N=4
N = 4
a = np.zeros(N + 1)
b = np.zeros(N + 1)

# 計算 a_k
for k in range(N + 1):
    a[k] = (2 / m) * np.sum(f_x * np.cos(2 * np.pi * k * x))

# 計算 b_k（k=1~N）
for k in range(1, N + 1):
    b[k] = (2 / m) * np.sum(f_x * np.sin(2 * np.pi * k * x))

# === 輸出完整的 S4(x) 式子 ===
terms = [f"{a[0]/2:.6f}"]
for k in range(1, N + 1):
    terms.append(f"{a[k]:+.6f} * cos({2*k}πx)")
    terms.append(f"{b[k]:+.6f} * sin({2*k}πx)")

S4_expression = "S₄(x) = " + " ".join(terms)

print("離散最小平方三角多項式 S₄(x)：\n")
print(S4_expression)
