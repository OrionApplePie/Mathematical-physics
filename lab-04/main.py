import itertools

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

from linalg import (
    seidel,
    norm_euclid,
    point_descent
)


EPS = 1E-5  # точность для м. Зейделя
NODES = 22  # количество узлов по стороне, 99 узлов считает очень долго
NN = NODES * NODES
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

    'none': 0.0,

    'bound_self': 2.0 + h**2 / 4.0,
    'bound_self_angle_r_two': 1.0 + h**2 / 6.0,
    'bound_self_angle_l_one': 1.0 + h**2 / 12.0,

    'bound_pair_vert_or_horz': h**2 / 24.0 - 0.5,
}

vals_f = {
    'bound_pair_vert_or_horz': h**2 / 2.0,
    'bound_self_angle_r_two': h**2 / 3.0,
    'bound_self_angle_l_one': h**2 / 6.0,

    'inner': h**2
}


def f_boundary_cond(node):
    """"Попадает ли узел в область где f-функция задается по другому."""
    x = h*node["i"]
    y = h*node["j"]
    if (0 <= x <= 0.5) and (0 <= y <= 0.5):
        return -6.00001
    return 2.0


def is_bound_node_index(nodes_list, i):
    # for nodes on bound
    for node in nodes_list:
        if node["node"] == i:
            if node["on_bound"]:
                return True
    return False


def get_type_of_node(node):
    """"""
    i = node['i']
    j = node['j']
    res = ''
    if (1 < i < NODES) and (1 < j < NODES):
        res = 'inner'
    elif (i == 1 and j == 1) or (i == NODES and j == NODES):
        res = 'bound_self_angle_r_two'
    elif (i == 1 and j == NODES) or (i == NODES and j == 1):
        res = 'bound_self_angle_l_one'
    else:
        res = 'bound_pair_vert_or_horz'
    return res


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

    # for nodes on bound
    if (
        i1 == i2 == 1 or
        i1 == i2 == NODES or
        j1 == j2 == 1 or
        j1 == j2 == NODES
    ):
        if ii == 0 and jj == 0:
            res = 'bound_self'
        if abs(ii) == 1 or abs(jj) == 1:
            res = 'bound_pair_vert_or_horz'
        if i1 == i2 == j1 == j2 == 1 or i1 == i2 == j1 == j2 == NODES:
            res = 'bound_self_angle_r_two'
        if (i1 == i2 == 1 and j1 == j2 == NODES) or (i1 == i2 == NODES and j1 == j2 == 1):
            res = 'bound_self_angle_l_one'
    return res


if __name__ == "__main__":
    node_num_count = 1
    nodes_list = []
    #  обходим все узлы и сохраняем в список вместе с координатами.
    is_on_bound = False

    for row_num in range(1, NODES + 1):
        for col_num in range(1, NODES + 1):
            if ((1 <= col_num <= NODES) and row_num == 1) or \
                 ((1 <= col_num <= NODES) and row_num == NODES) or \
                 ((1 <= row_num <= NODES) and col_num == 1) or \
                 ((1 <= row_num <= NODES) and col_num == NODES):
                is_on_bound = True

            nodes_list.append(
                {
                    'node': node_num_count,
                    'i': col_num,  # x
                    'j': row_num,  # y
                    'on_bound': is_on_bound
                }
            )
            is_on_bound = False
            node_num_count += 1

    # for node in nodes_list:
    #     print("node_number: {0}, is on bound: {1}".format(
    #         node['node'], node['on_bound']))
    # матрица жесткости
    # правая часть
    f = np.array(
        [0.0]*NN
    )

    matrix = np.zeros((NN, NN))
    # обходим каждый узел с каждым и заполняем матрицу жесткости
    # TODO: использовать симметричность матрицы
    for node1, node2 in itertools.product(nodes_list, repeat=2):

        a_ki = VALS[
            get_type_of_pair(node1, node2)
        ]
        # print("node1 {} --> node2 {} --> {} = {}, {} --> f = {}".format(
        #     node1['node'],
        #     node2['node'],
        #     get_type_of_pair(node1, node2),
        #     a_ki,
        #     get_type_of_node(node1),
        #     vals_f[get_type_of_node(node1)]
        # ))
        f_cond = f_boundary_cond(node1)

        f[node1['node'] - 1] = f_cond * vals_f[get_type_of_node(node1)]

        matrix[node1['node'] - 1][node2['node'] - 1] = a_ki
    
    # print(f)
    # print(vals_f)

    # решение слау встроенным решателем
    # sol = np.linalg.solve(matrix, f)
    # решение слау методом Зейделя
    # sol = seidel(
    #     matrix,
    #     f,
    #     EPS,
    #     norm_euclid
    # )
    sol = point_descent(
        matrix,
        f,
        EPS,
        norm_euclid,
        nodes_list,
        is_bound_node_index
    )

    # sol = np.around(sol, decimals=3)

    # print(sol)

    # из вектора решения делаем двумерный вектор
    # размером NODES на NODES
    Z = np.reshape(sol, (NODES, NODES))
    # создание сетки
    X = np.arange(0.0, 1.0, h)
    Y = np.arange(0.0, 1.0, h)
    X, Y = np.meshgrid(X, Y)

    # график поверхности 3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        linewidth=1,
        antialiased=True
    )

    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.0001f'))

    # шкала - бар
    # fig.colorbar(
    #     mappable=surf,
    #     shrink=0.5,
    #     aspect=5
    # )

    plt.show()
