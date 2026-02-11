import numpy as np

# 2.1) Функція та її градієнт методом Гессе (бо "аналітично" розписав у jpg файлі, суто щоб не повторюватись)    
def f(v):
    x, y = v
    return x**2 + x*y + y**2 - 6*x - 9*y + 20

def grad_f(v):
    x, y = v
    # df/dx = 2x + y - 6
    # df/dy = x + 2y - 9
    return np.array([2*x + y - 6, x + 2*y - 9], dtype=float)

def hessian_f(v=None):
    # d2f/dx2 = 2, d2f/dxdy = 1, d2f/dy2 = 2
    return np.array([[2.0, 1.0],
                     [1.0, 2.0]])

# 2.2) Критична точка, розв'язуємо дельта(f) = 0
# 2x + y - 6 = 0
# x + 2y - 9 = 0
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
b = np.array([6.0, 9.0])
x_star, y_star = np.linalg.solve(A, b)
xystar = np.array([x_star, y_star])

H = hessian_f()
eigvals = np.linalg.eigvals(H)

print("Аналітична критична точка (x*, y*):", xystar)
print("f(x*, y*):", f(xystar))
print("Гессе H:\n", H)
print("Власні значення H:", eigvals)
print("H позитивно визначена? ->", np.all(eigvals > 0))

#  2.3-2.4) Чисельна перевірка (BFGS) з різних екстремумів 
from scipy.optimize import minimize

starts = [
    np.array([0.0, 0.0]),
    np.array([10.0, 10.0]),
    np.array([-5.0, 15.0]),
]

print("\n--- BFGS оптимізація ---")
for s in starts:
    res = minimize(f, s, method="BFGS", jac=grad_f)
    print(f"Start {tuple(s)} -> x = {res.x}, f = {res.fun:.10f}, iters = {res.nit}, success = {res.success}")

#----------------------------------------------------------------------------------------------
# Висновок: 2.1 (x*, y*) = ( 1, 4) 
# 2.2 f(1;4) = 3
# 2.3 Власні значення Гессе: 3 і 1 -> обидва > 0, отже це - Cтрогий локальний мінімум для квадратичної функції!
