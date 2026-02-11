from dataclasses import dataclass

# --------------- Модель задачі ---------------

def Q(x: float) -> float:
    """Попит (кількість продажів на місяць)"""
    return 2000 - 0.8 * x

def P(x: float) -> float:
    """Прибуток"""
    return (x - 800) * Q(x)


# ---------------- 1.1) Аналітично --------------      # Це щоб не розписувати на бумазі
 
def analytical_optimum():
    # Розкриємо дужки:
    # P(x) = (x-800)(2000-0.8x)
    #      = -0.8x^2 + 2640x - 1_600_000
    # P'(x) = -1.6x + 2640
    # P'(x)=0 => x* = 2640/1.6 = 1650
    x_star = 2640 / 1.6

    # P''(x) = -1.6 < 0 => максимум
    profit = P(x_star)
    demand = Q(x_star)
    return x_star, demand, profit


# -------------- 1.2-1.3) Чисельно через SciPy -------------

def numeric_optimum_scipy():
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as e:
        raise SystemExit(
            "SciPy не встановлено. Встанови: pip install scipy\n"
            f"Деталі: {e}"
        )

    # minimize_scalar шукає мінімум, тому мінімізуємо -P(x)
    res = minimize_scalar(lambda x: -P(x), bounds=(800, 2500), method="bounded")

    x_opt = res.x
    demand = Q(x_opt)
    profit = P(x_opt)
    return res, x_opt, demand, profit


# -------------- Tasting ---------------

def main():
    x_a, q_a, p_a = analytical_optimum()
    print("=== Аналітичний розв'язок ===")
    print(f"x* = {x_a:.6f} грн")
    print(f"Q(x*) = {q_a:.6f} од./міс")
    print(f"P(x*) = {p_a:.2f} грн/міс")
    print("Тип екстремуму: максимум (бо P''(x) = -1.6 < 0)\n")

    res, x_n, q_n, p_n = numeric_optimum_scipy()
    print("=== Чисельний розв'язок (SciPy minimize_scalar) ===")
    print(f"x_opt = {x_n:.6f} грн")
    print(f"Q(x_opt) = {q_n:.6f} од./міс")
    print(f"P(x_opt) = {p_n:.2f} грн/міс")
    print(f"success = {res.success}, nfev = {res.nfev}\n")

    print("=== Порівняння ===")
    print(f"Різниця в ціні: |x_a - x_n| = {abs(x_a - x_n):.12f}")
    print(f"Різниця в прибутку: |P_a - P_n| = {abs(p_a - p_n):.12f}")


if __name__ == "__main__":
    main()


# -------------- 1.4) Висновки + порівняння ------------
#  х' = 1650 (грн)
#  попит Q(x') = 680 (од\міс)
#  максимальний прибуток P(х') = 578000 (грн)
# Вийшло трохи заплутаніше ніж звичайно, але я так зробив бо було б ще важче розбити на аналітучну частину, а роздільно ще й програмну... 0_0
