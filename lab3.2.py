import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Определяем функцию f(x)
def f(x):
    return 1/3 + np.cos(10 + 2.3**np.abs(x))

# Определяем многочлены Чебышёва первого рода
def chebyshev(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return 2 * x * chebyshev(n - 1, x) - chebyshev(n - 2, x)

# Преобразование x из [a, b] в [-1, 1]
def transform(x, a, b):
    return (2 * x - (b + a)) / (b - a)

# Функция для вычисления матрицы скалярных произведений
def compute_inner_products(a, b, m):
    inner_products = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            inner_products[i][j] = quad(lambda x: chebyshev(i, transform(x, a, b)) * chebyshev(j, transform(x, a, b)) / np.sqrt(1 - transform(x, a, b)**2), a, b)[0]
    return inner_products

# Функция для вычисления вектора скалярных произведений
def compute_vector(a, b, m):
    vector = np.zeros(m + 1)
    for i in range(m + 1):
        vector[i] = quad(lambda x: f(x) * chebyshev(i, transform(x, a, b)) / np.sqrt(1 - transform(x, a, b)**2), a, b)[0]
    return vector

# Основная функция для нахождения коэффициентов α
def find_coefficients(a, b, m):
    inner_products = compute_inner_products(a, b, m)
    vector = compute_vector(a, b, m)
    
    # Решаем систему уравнений
    alpha = np.linalg.solve(inner_products, vector)
    return alpha

# Параметры интегрирования
a = -1
b = 2.5

# Подбираем подходящее значение m из предыдущего пункта
m = 5 # Предполагаем m из предыдущего пункта

# Находим коэффициенты α для многочленов Чебышёва
alpha = find_coefficients(a, b, m)

# Функция g(x) на основе найденных коэффициентов
def g(x):
    return sum(alpha[k] * chebyshev(k, transform(x, a, b)) for k in range(m + 1))

# Вычисление ошибки E
error_integral = quad(lambda x: (f(x) - g(x))**2, a, b)[0]
print(f"Ошибка ɛ с использованием многочленов Чебышёва: {error_integral}")

# Визуализация результатов
x_vals = np.linspace(a, b, 400)
f_vals = f(x_vals)
g_vals = g(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, f_vals, label='f(x)', color='blue')
plt.plot(x_vals, g_vals, label='g(x)', color='red', linestyle='--')
plt.title(f'Сравнение функции f(x) и её приближения g(x) при m={m} (многочлены Чебышёва)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()