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

N = 20  # количество узлов по стороне
NODES = N * N  # всего узлов
h = 1.0 / N

E = 200E3  # aluminium
Mu = 0.3
Mu_1 = Mu / (1 - Mu)
E_1 = E / (1 - Mu*Mu)
G = G1 = E / (2*(1 + Mu))

P1 = -1000
P2 = -100

# константы мапперы со значениями интегралов различных комбинаций
# TODO: добавить картинку схему узлов и обозначений

# dphi_ik/dx1 * dphi_i/dx1
vals_dx1_dx1 = {
    # внутренние (неграничные) пары узлов
    'self_inner': 2.0,

    'up': 0.0,
    'down': 0.0,

    'left': -1.0,
    'right': -1.0,

    'right_up': 0.0,
    'left_down': 0.0,

    # узел сам с собой на границе
    'bound_self_triangle_one': 0.5,  # и верхний и нижний - интегралы одинаковые
    'bound_self_triangle_two': 0.5,

    'bound_self_vert_left': 1,
    'bound_self_vert_right': 1,

    'bound_self_horz_up': 1,
    'bound_self_horz_down': 1,

    # пары узлов на границе
    'bound_pair_horz_up': -0.5,
    'bound_pair_horz_down': -0.5,

    'bound_pair_vert_left': 0,
    'bound_pair_vert_right': 0,

    'none': 0.0,  # если узлы не имеют общийх областей
}

# dphi_ik/dx2 * dphi_i/dx1
vals_dx2_dx1 = {
    # внутренние (неграничные) пары узлов
    'self_inner': -1,

    'up': 0.5,
    'down': 0.5,

    'left': 0.5,
    'right': 0.5,

    'right_up': -0.5,
    'left_down': -0.5,

    # узел сам с собой на границе
    'bound_self_triangle_one': -0.5,  # и верхний и нижний - интегралы одинаковые
    'bound_self_triangle_two': 0.0,

    'bound_self_vert_left': -0.5,
    'bound_self_vert_right': -0.5,

    'bound_self_horz_up': -0.5,
    'bound_self_horz_down': -0.5,

    # пары узлов на границе
    'bound_pair_horz_up': 0.5,
    'bound_pair_horz_down': 0,

    'bound_pair_vert_left': 0.5,
    'bound_pair_vert_right': 0,

    'none': 0.0,  # если узлы не имеют общийх областей
}

# dphi_ik/dx2 * dphi_i/dx1
vals_dx2_dx2 = {
    # внутренние (неграничные) пары узлов
    'self_inner': 2,

    'up': -1,
    'down': -1,

    'left': 0.0,
    'right': 0.0,

    'right_up': 0.0,
    'left_down': 0.0,

    # узел сам с собой на границе
    'bound_self_triangle_one': 0.5,  # и верхний и нижний - интегралы одинаковые
    'bound_self_triangle_two': 0.5,

    'bound_self_vert_left': 1,
    'bound_self_vert_right': 1,

    'bound_self_horz_up': 1,
    'bound_self_horz_down': 1,

    # пары узлов на границе
    'bound_pair_horz_up': 0,
    'bound_pair_horz_down': 0,

    'bound_pair_vert_left': -0.5,
    'bound_pair_vert_right': -0.5,

    'none': 0.0,  # если узлы не имеют общийх областей
}

# dphi_ik/dx2 * dphi_i/dx1
vals_dx1_dx2 = {
    # внутренние (неграничные) пары узлов
    'self_inner': -1,

    'up': 0.5,
    'down': 0.5,

    'left': 0.5,
    'right': 0.5,

    'right_up': -0.5,
    'left_down': -0.5,

    # узел сам с собой на границе
    'bound_self_triangle_one': -0.5,  # и верхний и нижний - интегралы одинаковые
    'bound_self_triangle_two': 0.0,

    'bound_self_vert_left': -0.5,
    'bound_self_vert_right': -0.5,

    'bound_self_horz_up': -0.5,
    'bound_self_horz_down': -0.5,

    # пары узлов на границе
    'bound_pair_horz_up': 0.0,
    'bound_pair_horz_down': 0.5,

    'bound_pair_vert_left': 0.0,
    'bound_pair_vert_right': 0.5,

    'none': 0.0,  # если узлы не имеют общийх областей
}


def get_type_of_node(node):
    """Определение типа узла."""
    i = node['i']
    j = node['j']

    res = ''

    if (1 < i < N) and (1 < j < N):
        res = 'inner'  # неграничный внутренний узел

    elif (i == 1 and j == 1) or (i == N and j == N):
        # левый нижний или верхний правый (2 кусочка треугольника)
        res = 'bound_angle_r_two'

    elif (i == 1 and j == N) or (i == N and j == 1):
        # левый верхний или нижний правый (1 кусочек треугольника)
        res = 'bound_angle_l_one'

    elif (i == 1 and j != 1 and j != N):
        # на левой вертикали
        res = 'bound_vert_left'

    elif (i == N and j != 1 and j != N):
        # на правой вертикили
        res = 'bound_vert_right'

    elif (j == 1 and i != 1 and i != N):
        # нижняя горизонталь
        res = 'bound_horz_down'

    elif (j == N and i != 1 and i != N):
        # верхняя горизонталь
        res = 'bound_horz_up'

    else:
        raise ValueError

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

    node1_type = get_type_of_node(node1)
    node2_type = get_type_of_node(node2)
    # print(f"{i1=} {i2=} {j1=} {j2=}")
    res = ''

    if ii == 0 and jj == 0:
        res = 'self_inner'
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
        res = 'none'  # нет общих областей - интеграл равен 0

    # если на границе
    if (
        i1 == i2 == 1 or
        i1 == i2 == N or
        j1 == j2 == 1 or
        j1 == j2 == N
    ):
        if (ii == 0 and jj == 0):  # если узел с самим собой
            if node1_type == 'bound_angle_l_one':
                res = 'bound_self_triangle_one'

            elif node1_type == 'bound_angle_r_two':
                res = 'bound_self_triangle_two'

            elif node1_type == 'bound_vert_left':
                res = 'bound_self_vert_left'

            elif node1_type == 'bound_vert_right':
                res = 'bound_self_vert_right'

            elif node1_type == 'bound_horz_up':
                res = 'bound_self_horz_up'

            elif node1_type == 'bound_horz_down':
                res = 'bound_self_horz_down'

            else:
                raise ValueError  # TODO:fix me!!!

        # проверка - если сосед
        elif abs(ii) == 1 and j1 == 1:
            res = 'bound_pair_horz_down'

        elif abs(ii) == 1 and j1 == N:
            res = 'bound_pair_horz_up'

        elif abs(jj) == 1 and i1 == 1:
            res = 'bound_pair_vert_left'

        elif abs(jj) == 1 and i1 == N:
            res = 'bound_pair_vert_right'

        else:
            res = 'none'

    return res


def main():
    node_num_count = 1
    nodes_list = []
    # обходим все узлы и сохраняем в список вместе с координатами,
    # координаты (j, i) - номер строки, номер колонки
    # обход низу вверх, слева направо
    for row_num in range(1, N + 1):
        for col_num in range(1, N + 1):
            nodes_list.append(
                {
                    'node_num': node_num_count,
                    'i': col_num,
                    'j': row_num
                }
            )
            node_num_count += 1

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
        pair_type = get_type_of_pair(node1, node2)

        i = node1['node_num']
        j = node2['node_num']

        node1_type = get_type_of_node(node1)
        if node1_type == 'bound_vert_left':
            f_upper[i] = P1

        if node1_type == 'bound_vert_right':
            f_lower[i] = P2

        # if node1_type == 'bound_horz_down':
        #     f_lower[i] = 0

        # if node1_type == 'bound_horz_up':
        #     f_upper[i] = 0

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

        # print(
        #     "node1 {0} --> node2 {1}, pair type: {2}".format(
        #         node1['node_num'],
        #         node2['node_num'],
        #         pair_type
        #     )
        # )

    factor1 = E_1 / (1.0 - Mu_1*Mu_1)

    t1_1 *= factor1
    t2_2 *= factor1*Mu_1
    t1_3 *= G
    t2_4 *= G

    t1_5 *= factor1*Mu_1
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
    # решение слау встроенным решателем
    sol = np.linalg.solve(k_matrix, f)

    print(f"solution shape: {sol.shape}")
    print(f"nodes: {NODES}, N={N}")

    X, Y = np.meshgrid(np.arange(0, 1, h), np.arange(0, 1, h))

    U = np.reshape(sol[:NODES], (N, N))
    V = np.reshape(sol[NODES:], (N, N))

    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V, units='xy', scale=2, color='red')

    ax.set_aspect('equal')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.title('Поле перемещений', fontsize=10)

    # plt.savefig('how_to_plot_a_vector_field_in_matplotlib_fig1.png',
    # bbox_inches='tight')
    plt.show()
    plt.close()

    # # из вектора решения делаем двумерный вектор
    # # размером NODES на NODES
    # Z = np.reshape(sol, (NODES, NODES))
    # # создание сетки
    # X = np.arange(0.0, 1.0, h)
    # Y = np.arange(0.0, 1.0, h)
    # X, Y = np.meshgrid(X, Y)

    # # график поверхности 3d
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(
    #     X, Y, Z,
    #     cmap=cm.coolwarm,
    #     linewidth=1,
    #     antialiased=True
    # )

    # # ax.zaxis.set_major_formatter(FormatStrFormatter('%.0001f'))

    # # шкала - бар
    # # fig.colorbar(
    # #     mappable=surf,
    # #     shrink=0.5,
    # #     aspect=5
    # # )

    # plt.show()


if __name__ == "__main__":
    main()
