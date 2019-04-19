import itertools

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

from linalg import (
    seidel,
    norm_avg_square
)


EPS = 1E-5  # точность для м. Зейделя
NODES = 99  # количество узлов по стороне, 99 узлов считает очень долго
h = 1.0 / NODES
# константа маппер со значениями элементов м. жесткости (интегралы)
VALS = {
    'self': 4.0 + h**2 / 2.0,

    'up': h**2 / 12.0 - 1.0,
    'down': h**2 / 12.0 - 1.0,

    'right': h**2 / 12.0 - 1.0,
    'left': h**2 / 12.0 - 1.0,

    'right_up': h**2 / 12.0,
    'left_down': h**2 / 12.0,

    'none': 0.0
}


def get_type_of_pair(node1, node2):
    """Функция вычисляет 'тип соседства' 2х узлов.
    Работает только для квадратной области."""
    i1 = node1['i']
    i2 = node2['i']
    j1 = node1['j']
    j2 = node2['j']

    ii = i1 - i2
    jj = j1 - j2

    res = ''
    if ii == 0 and jj == 0:
        res = 'self'
    elif ii == 1 and jj == 1:
        res = 'left_down'
    elif ii == -1 and jj == -1:
        res = 'right_up'
    elif ii == 0 and jj == 1:
        res = 'down'
    elif ii == 0 and jj == -1:
        res = 'up'
    elif ii == -1 and jj == 0:
        res = 'right'
    elif ii == 1 and jj == 0:
        res = 'left'
    else:
        res = 'none'

    return res


if __name__ == "__main__":    
    node_num_count = 1
    nodes_list = []
    #  обходим все узлы и сохраняем в список вместе с координатами.
    for row_num in range(1, NODES + 1):
        for col_num in range(1, NODES + 1):
            nodes_list.append(
                {
                    'node': node_num_count,
                    'i': col_num,
                    'j': row_num
                }
            )
            node_num_count += 1
    # матрица жесткости
    matrix = np.zeros((NODES*NODES, NODES*NODES))
    # обходим каждый узел с каждым и заполняем матрицу жесткости
    # TODO: использовать симметричность матрицы
    for node1, node2 in itertools.product(nodes_list, repeat=2):
        pair_type = VALS[get_type_of_pair(node1, node2)]
        matrix[node1['node'] - 1][node2['node'] - 1] = pair_type

    # правая часть
    f = np.array(
        [h ** 2]*(NODES*NODES)
    )
    # решение слау встроенным решателем
    # sol = np.linalg.solve(matrix, f)
    # решение слау методом Зейделя
    sol = seidel(
        matrix,
        f,
        EPS,
        norm_avg_square
    )

    # из вектора решения делаем двумерный вектор
    # размером NODES на NODES
    Z = np.reshape(sol, (NODES, NODES))
    # создание сетки
    X = np.arange(0, 1, h)
    Y = np.arange(0, 1, h)
    X, Y = np.meshgrid(X, Y)

    # график поверхности 3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=True
    )
    
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # шкала - бар
    fig.colorbar(
        mappable=surf,
        shrink=0.5,
        aspect=5
    )

    plt.show()
