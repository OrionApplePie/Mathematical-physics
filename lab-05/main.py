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

from linalg import norm_euclid, seidel

EPS = 1E-5  # точность для м. Зейделя
NODES = 4 # количество узлов по стороне, 99 узлов считает очень долго
NN = NODES * NODES
h = 1.0 / NODES

# константы мапперы со значениями интегралов различных комбинаций

vals_f = {
    'bound_pair_vert_or_horz': h**2 / 2.0,
    'bound_self_angle_r_two': h**2 / 3.0,
    'bound_self_angle_l_one': h**2 / 6.0,

    'inner': h**2,
    "error": 42
}
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
    'bound_pair_vert_left': 0,
    'bound_pair_vert_right': 0.5,

    'none': 0.0,  # если узлы не имеют общийх областей
}

# dphi_ik/dx2 * dphi_i/dx1
vals_dx2_dx2 = {
    # внутренние (неграничные) пары узлов 
    'self_inner': 2,

    'up': -1,
    'down': -1,

    'left': 0.5,
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
    'bound_pair_horz_up': 0.0,
    'bound_pair_horz_down': 0.0,
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
    'bound_pair_horz_up': 0.5,
    'bound_pair_horz_down': 0,
    'bound_pair_vert_left': 0.5,
    'bound_pair_vert_right': 0,

    'none': 0.0,  # если узлы не имеют общийх областей
}


def get_type_of_node(node):
    """Определение типа узла."""
    i = node['i']
    j = node['j']

    res = ''

    if (1 < i < NODES) and (1 < j < NODES):
        res = 'inner' # неграничный внутренний узел

    elif (i == 1 and j == 1) or (i == NODES and j == NODES):
        # левый нижний или верхний правый (2 кусочка треугольника)
        res = 'bound_angle_r_two'

    elif (i == 1 and j == NODES) or (i == NODES and j == 1):
        # левый верхний или нижний правый (1 кусочек треугольника)
        res = 'bound_angle_l_one'

    elif (i == 1 and j != 1 and j != NODES):
        # на левой вертикали
        res = 'bound_vert_left'

    elif (i == NODES and j != 1 and j != NODES):
        # на правой вертикили
        res = 'bound_vert_right'

    elif (j == 1 and i != 1 and i != NODES):
        # нижняя горизонталь
        res = 'bound_horz_down'

    elif (j == NODES and i != 1 and i != NODES):
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
        res = 'none' # нет общих областей - интеграл равен 0

    # если на границе
    if (
        i1 == i2 == 1 or
        i1 == i2 == NODES or
        j1 == j2 == 1 or
        j1 == j2 == NODES
    ):
        if (ii == 0 and jj == 0): # если узел с самим собой
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
                raise ValueError # TODO:fix me!!!

        # проверка - если сосед
        elif abs(ii) == 1 and j1 == 1:
            res = 'bound_pair_horz_down'

        elif abs(ii) == 1 and j1 == NODES:
            res = 'bound_pair_horz_up'
        
        elif abs(jj) == 1 and i1 == 1:
            res = 'bound_pair_vert_left'

        elif abs(jj) == 1 and i1 == NODES:
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
    for row_num in range(1, NODES + 1):
        for col_num in range(1, NODES + 1):
            nodes_list.append(
                {
                    'node_num': node_num_count,
                    'i': col_num,
                    'j': row_num
                }
            )
            node_num_count += 1

    # матрица жесткости
    for node in nodes_list:
        print((f"node #{node['node_num']}: row={node['j']}, col={node['i']}\n"
               f"type: {get_type_of_node(node)}\n"
               f"----------------------------"))
    # правая часть
    f = np.array(
        [0.0]*NN
    )

    matrix = np.zeros((NN, NN))
    # обходим каждый узел с каждым и заполняем матрицу жесткости
    # TODO: использовать симметричность матрицы
    t1_1 = []
    t2_1 = []
    t1 = []
    t2 = []
    t3 = []
    t4 = []

    for node1, node2 in itertools.product(nodes_list, repeat=2):

        pair_type = get_type_of_pair(node1, node2)

        t1.append(vals_dx1_dx1[pair_type])
        t2.append(vals_dx2_dx1[pair_type])
        t3.append(vals_dx2_dx2[pair_type])
        t4.append(vals_dx1_dx2[pair_type])

        print(
            "node1 {0} --> node2 {1}, pair type: {2}, val={3}".format(
                node1['node_num'],
                node2['node_num'],
                pair_type,
                a_ki
            )
        )
        # f[node1['node'] - 1] = vals_f[get_type_of_node(node1)]

        # matrix[node1['node'] - 1][node2['node'] - 1] = a_ki
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
    # sol = np.around(sol, decimals=3)

    # print(sol)
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
