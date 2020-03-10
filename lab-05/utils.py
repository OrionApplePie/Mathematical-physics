
def create_nodes(N=5):
    node_num_count = 1
    nodes_list = []
    # обходим все узлы и сохраняем в список вместе с координатами,
    # координаты (j, i) - номер строки, номер колонки
    # обход низу вверх, слева направо
    for row_num in range(1, N + 1):
        for col_num in range(1, N + 1):
            # пропускаем узлы за закрепоенной границе
            if 1 > 2:  # (1 <= col_num <= N) and (row_num == 1 or row_num == N):
                continue
            else:
                nodes_list.append(
                    {
                        'node_num': node_num_count,
                        'i': col_num,
                        'j': row_num
                    }
                )
                node_num_count += 1
    return nodes_list


def get_type_of_node(node, N):
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


def get_type_of_pair(node1, node2, N):
    """Функция вычисляет 'тип соседства' 2х узлов.
    Работает только для квадратной области."""
    i1 = node1['i']
    i2 = node2['i']
    j1 = node1['j']
    j2 = node2['j']

    ii = i1 - i2
    jj = j1 - j2

    node1_type = get_type_of_node(node1, N)
    node2_type = get_type_of_node(node2, N)

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
