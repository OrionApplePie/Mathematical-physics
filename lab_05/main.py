"""
Плоская задача теории упругости. Область - единичный квадрат.
Верхняя и нижняя стороны закреплены, на левую и правую действуют
поверхностные силы P,
объемные силы f = (0, 0)
"""
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from lab_05.integrals import (
    vals_dx1_dx1,
    vals_dx1_dx2,
    vals_dx2_dx1,
    vals_dx2_dx2
)
from lab_05.utils import (
    create_nodes,
    get_type_of_node,
    get_type_of_pair
)


def main():
    N = 21  # количество узлов по стороне
    NODES = N * (N - 2)  # всего узлов (без закрепленных границ)
    h = 1.0 / (N - 1)

    E = 2.1e6
    Mu = 0.3
    G = E / (2. * (1. + Mu))
    factor1 = E / (1. - Mu * Mu)

    r_part = (h - h / 2.) * (1. + (1. + 1. / h) ** 0.5)
    P_left = (10000, 0)
    P_right = (-10000, 0)

    nodes_list = create_nodes(N)

    # матрица жесткости
    # for node in nodes_list:
    #     print((f"node #{node['node_num']}: row={node['j']}, col={node['i']}\n"
    #            f"type: {get_type_of_node(node)}\n"
    #            f"----------------------------"))

    # правая часть
    f_upper = np.zeros(NODES)
    f_lower = np.zeros(NODES)

    # По порядку как в уравнениях (3)
    # первый индекс - это верхний у t^(i)
    # в уравнеиях надо поправить, чтобы шло t1, t2 - нужно для решения слау

    t1_1 = np.zeros((NODES, NODES))
    t2_2 = np.zeros((NODES, NODES))

    t1_3 = np.zeros((NODES, NODES))
    t2_4 = np.zeros((NODES, NODES))

    t1_5 = np.zeros((NODES, NODES))
    t2_6 = np.zeros((NODES, NODES))

    t1_7 = np.zeros((NODES, NODES))
    t2_8 = np.zeros((NODES, NODES))

    # обходим каждый узел с каждым и заполняем матрицу жесткости
    # TODO: использовать портрет матрицы?
    for node1, node2 in combinations_with_replacement(nodes_list, r=2):
        pair_type = get_type_of_pair(node1, node2, N)

        i = node1['node_num']
        j = node2['node_num']

        val_x1x1 = vals_dx1_dx1[pair_type]
        val_x2x1 = vals_dx2_dx1[pair_type]
        val_x2x2 = vals_dx2_dx2[pair_type]
        val_x1x2 = vals_dx1_dx2[pair_type]

        t1_1[i-1][j-1] = val_x1x1
        t2_2[i-1][j-1] = val_x2x1

        t1_3[i-1][j-1] = val_x2x2
        t2_4[i-1][j-1] = val_x1x2

        t1_5[i-1][j-1] = val_x1x2  # изменен порядок след.
        t2_6[i-1][j-1] = val_x2x2

        t1_7[i-1][j-1] = val_x2x1
        t2_8[i-1][j-1] = val_x1x1

        # и симметричные эл.

        t1_1[j-1][i-1] = val_x1x1
        t2_2[j-1][i-1] = val_x2x1

        t1_3[j-1][i-1] = val_x2x2
        t2_4[j-1][i-1] = val_x1x2

        t1_5[j-1][i-1] = val_x1x2  # изменен порядок след.
        t2_6[j-1][i-1] = val_x2x2

        t1_7[j-1][i-1] = val_x2x1
        t2_8[j-1][i-1] = val_x1x1

        node1_type = get_type_of_node(node1, N)

        if node1_type == 'bound_vert_left':
            f_upper[i-1] = P_left[0]

        if node1_type == 'bound_vert_right':
            f_upper[i-1] = P_right[0]

        # info = """node1: {0}, node2: {1},
        #           node1 type: {2},
        #           node2 type: {3},
        #           pair type: {4}""".format(
        #     node1['node_num'],
        #     node2['node_num'],
        #     get_type_of_node(node1, N),
        #     get_type_of_node(node2, N),
        #     pair_type
        # )

        # print(info)

    t1_1 *= factor1
    t2_2 *= factor1 * Mu
    t1_3 *= G
    t2_4 *= G

    t1_5 *= factor1 * Mu
    t2_6 *= factor1
    t1_7 *= G
    t2_8 *= G

    # приведение подобных
    t1_1 += t1_3
    t2_2 += t2_4

    t1_5 += t1_7
    t2_6 += t2_8

    part_upper = np.concatenate((t1_1, t2_2), axis=1)
    part_lower = np.concatenate((t1_5, t2_6), axis=1)

    k_matrix = np.concatenate((part_upper, part_lower))
    f = np.concatenate((f_upper, f_lower))
    print("start slae solving...")
    # решение слау
    sol = np.linalg.solve(k_matrix, f)
    n, = sol.shape

    X, Y = np.meshgrid(
        np.linspace(0, 1, N),
        np.linspace(0, 1, N)
    )

    U, V = np.split(sol, 2)

    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V, units='xy', scale=2, color='red')

    ax.set_aspect('equal')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.title('Поле перемещений', fontsize=10)

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
