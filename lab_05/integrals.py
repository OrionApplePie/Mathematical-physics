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
