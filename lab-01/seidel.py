import math
import numpy as np


def gen_three_diag_by_3(elements, n):
    """Генерация трехдиагональной матрицы размером n x n на основе 3 элементов."""
    left, diag, right = elements
    res = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                res[i, j] = diag
                if j - 1 >= 0:
                    res[i, j - 1] = left 
                if j + 1 < n:
                    res[i, j + 1] = right
    
    return res


def norm1(vec1, vec2):
    """Норма максимум по абс. знач. компонента вектора."""
    n = len(vec1)
    return max(
        abs(vec1[i] - vec2[i]) for i in range(n)
    )


def norm_euclid(vec1, vec2):
    """Евклидова норма."""
    n = len(vec1)

    return math.sqrt(
        sum(
            (vec1[i] - vec2[i]) ** 2 for i in range(n)
        )
    )


def norm_avg_square(vec1, vec2):
    """Среднеквадратическая норма."""
    n = len(vec1)

    return math.sqrt(
        sum(
            (vec1[i] - vec2[i]) ** 2 for i in range(n)
        ) / n
    )



def seidel1(A, b, eps):
    """Реализация метода Зейделя из википедии."""
    n = len(A)
    x = [.0 for i in range(n)]

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = math.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new

    return x


def seidel(A, f, eps, norm):
    """
    Метод Зейделя.
    A - матрица,
    f -  вектор правой части,
    eps - заданная точность
    """
    m = len(f)    
    x_prev = np.zeros(m)
    x_next = np.zeros(m)

    while(True):
    # основной цикл итераций
        for i in range(m): # для каждой компоненты вектора решения (x)
            sum1 = 0
            sum2 = 0
            # Суммирование от 0 до i-1 -го элемента, или по всем эл-м от начального (0) которые меньше i-го
            # range(i) дает элементы от 0 до i-1 (!)
            for j in range(i):
                sum1 += A[i, j]*x_next[j]
            # Суммирование от i+1-го до m-1-го элемента, или по всем эл-м от начального (i+i) которые больше i-го
            # range(i+1, m) дает элементы от i+1 до m-1-го (!)
            for j in range(i+1, m):
                sum2 += A[i, j]*x_prev[j]
            # вычисление i-той компоненты вектора на текущем n+1 шаге
            x_next[i] = (f[i] - sum1 - sum2) / A[i, i]

        # if norm1(x_next - x_prev) < eps:
        if norm(x_next, x_prev) <= eps:
            # если по норме разность векторов решений на предыдущем и текущем шагах меньше эпс, то выход из главного цикла
            break
        # если нет - то текущее решение становиться предыдущим для новой итерации
        x_prev = np.copy(x_next)
        
    return x_next
