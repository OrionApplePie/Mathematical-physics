"""
Плоская задача теории упругости. Область - единичный квадрат.
Верхняя и нижняя стороны закреплены, на левую и правую действуют
поверхностные силы P, 
объемные силы f = 0
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from integrals import vals_dx1_dx1, vals_dx1_dx2, vals_dx2_dx1, vals_dx2_dx2
from utils import create_nodes, get_type_of_node, get_type_of_pair


def main():
    N = 25  # количество узлов по стороне
    
    NODES = N * (N-2)  # всего узлов (без закрепленных границ)
    h = 1.0 / (N-1)
    
    E = 700  # aluminium
    Mu = 0.04
    Mu_1 = Mu / (1 - Mu)
    E_1 = E / (1 - Mu*Mu)
    G = G1 = E / (2*(1 + Mu))
    
    r_part = (h - h/2)*(1 + (1 + 1/h) ** 0.5)
    print(f"{r_part}")
    
    P1 = 100*r_part
    P2 = -100*r_part

    nodes_list = create_nodes(N)

    # матрица жесткости
    # for node in nodes_list:
    #     print((f"node #{node['node_num']}: row={node['j']}, col={node['i']}\n"
    #            f"type: {get_type_of_node(node)}\n"
    #            f"----------------------------"))

    # правая часть
    f_upper = np.zeros(NODES)
    f_lower = np.zeros(NODES)
    # обходим каждый узел с каждым и заполняем матрицу жесткости
    # TODO: использовать портрет матрицы

    # По порядку как в уравнениях (3)
    # первый индекс - это верхний у t^(i)
    # в уравнеиях надо поправить чтобы шло t1, t2 - нужно для решения слау
    #
    t1_1 = np.zeros((NODES, NODES))
    t2_2 = np.zeros((NODES, NODES))

    t1_3 = np.zeros((NODES, NODES))
    t2_4 = np.zeros((NODES, NODES))

    t1_5 = np.zeros((NODES, NODES))
    t2_6 = np.zeros((NODES, NODES))

    t1_7 = np.zeros((NODES, NODES))
    t2_8 = np.zeros((NODES, NODES))

    for node1, node2 in itertools.product(nodes_list, repeat=2):
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

        t1_5[i-1][j-1] = val_x1x2
        t2_6[i-1][j-1] = val_x2x2

        t1_7[i-1][j-1] = val_x2x1
        t2_8[i-1][j-1] = val_x1x1

        node1_type = get_type_of_node(node1, N)

        if node1_type == 'bound_vert_left':
            f_upper[i-1] = P1
        if node1_type == 'bound_vert_right':
            f_lower[i-1] = P2

        # print(
        #     "node1 {0} --> node2 {1}, pair type: {2}".format(
        #         node1['node_num'],
        #         node2['node_num'],
        #         pair_type
        #     )
        # )

    factor1 = E / (1.0 - Mu*Mu)

    t1_1 *= factor1
    t2_2 *= factor1*Mu
    t1_3 *= G
    t2_4 *= G

    t1_5 *= factor1*Mu
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
    f = np.concatenate((f_lower, f_upper))
    # решение слау
    sol = np.linalg.solve(k_matrix, f)
    n, = sol.shape
    # sol = np.round(sol, 2)
    zer = np.zeros(N)

    X, Y = np.meshgrid(
        np.linspace(0, 1, N),
        np.linspace(0, 1, N)
    )

    V, U = np.split(sol, 2)

    U = np.concatenate((zer, U, zer))
    V = np.concatenate((zer, V, zer))

    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V, units='xy', scale=2, color='red')

    # ax.set_aspect('equal')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.title('Поле перемещений', fontsize=10)

    # plt.savefig('how_to_plot_a_vector_field_in_matplotlib_fig1.png',
    # bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
