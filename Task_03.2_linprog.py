import numpy as np
from scipy.optimize import linprog
                                                                  
# maximize 500x + 800y => minimize -(500x + 800y)  (за конспектом)
c = [-500, -800]

A_ub = [
    [2, 4],     # деревина
    [3, 2],    # робочий час
]
b_ub = [120, 90]

bounds = [(0, None), (0, None)]      # (це також з конспекту)

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

x, y = res.x
profit = 500*x + 800*y

print("x (стільці) =", x)
print("y (столи)   =", y)
print("Макс. прибуток =", profit)
print("Статус:", res.message)

