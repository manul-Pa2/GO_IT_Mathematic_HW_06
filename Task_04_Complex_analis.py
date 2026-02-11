import numpy as np
from scipy.optimize import approx_fprime, minimize_scalar, minimize
from scipy.integrate import quad

def P(t):
    return 100 + 40*t - 4*(t**2)

# 4.1) Похідна для часу t=2;5;8
def P_num_derivative(t, eps=1e-6):
    return approx_fprime(np.array([t], dtype=float),
                         lambda z: P(z[0]),
                         epsilon=eps)[0]

for t in [2, 5, 8]:
    dp = P_num_derivative(t)
    trend = "зростає" if dp > 0 else ("спадає" if dp < 0 else "пік/перелом")
    print(f"P'(t) при t={t}: {dp:.6f} -> продуктивність {trend}")

# 4.2) Пікова продуктивність P(max)
res_peak = minimize_scalar(lambda t: -P(t), bounds=(0, 10), method="bounded")
t_star = res_peak.x
P_star = P(t_star)
print("\nПік продуктивності:")
print("t* =", t_star)
print("P(t*) =", P_star)

# 4.3) Всього виробнгицтва
total_units, err = quad(P, 0, 10)
print("\nЗагальний обсяг виробництва за зміну:")
print("Integral =", total_units, "(похибка ~", err, ")")

def C(v):
    x, y = v
    return x**2 + y**2 - 10*x - 8*y + 50

# 4.4) Рахуємо похідну (x0,y0)
# 2x + y = 20
# x + 3y = 25
A = np.array([[2, 1],
              [1, 3]], dtype=float)
b = np.array([20, 25], dtype=float)

x0, y0 = np.linalg.solve(A, b)
print("\nПочаткові параметри з бюджету:")
print("x0 =", x0, "y0 =", y0)

# 4.5) Мінімізація BFGS (x0,y0)
res_min = minimize(C, x0=np.array([x0, y0]), method="BFGS")
x_opt, y_opt = res_min.x
C_min = res_min.fun

print("\nОптимізація витрат (BFGS):")
print("x* =", x_opt)
print("y* =", y_opt)
print("мінімальна вартість C* =", C_min)

# 4.6) Final: Total cost
total_cost = total_units * C_min
print("\nФінал:")
print("Загальна вартість =", total_cost)
