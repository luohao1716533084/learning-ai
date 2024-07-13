import numpy as np
import matplotlib.pyplot as plt

# 定义函数和导数
def f(x):
    return x**2 + np.sin(x)

def f_prime(x):
    return 2*x + np.cos(x)

# 生成x的值
x_values = np.linspace(-5, 5, 100)

# 绘制函数和导数的图像
plt.figure(figsize=(10, 6))
plt.plot(x_values, f(x_values), label='f(x)')
plt.plot(x_values, f_prime(x_values), label="f'(x)")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and its Derivative')
plt.legend()
plt.grid(True)
plt.show()
