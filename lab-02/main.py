import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

from linalg import (
    seidel,
    norm_avg_square
)


def get_type(node1, node2):
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


NODES = 12  # количество узлов по стороне

h = 1.0 / NODES

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

# for node in nodes_list:
#     print("node # {} -- > i={}, j={}".format(
#             node['node'],
#             node['i'],
#             node['j']
#         )
#     )

# матрица жесткости
matrix = np.zeros((NODES*NODES, NODES*NODES))

# константа маппер со значениями элементов м. жесткости (интегралы)
vals = {
    'self': 4.0 + ((h**2) / 2.0),

    'up': ((h**2) / 12.0) - 1.0,
    'down': ((h**2) / 12.0) - 1.0,

    'right': ((h**2) / 12.0) - 1.0,
    'left': ((h**2) / 12.0) - 1.0,

    'right_up': (h**2) / 12.0,
    'left_down': (h**2) / 12.0,

    'none': 0.0
}

# for node1 in nodes_list:
#     for node2 in nodes_list:
#         res = "{} + {} = {}".format(
#             node1['node'],
#             node2['node'],
#             get_type(node1, node2)
#         )
#         print(res)

# обходим узлы и заполняем матрицу жесткости
for node1 in nodes_list:
    for node2 in nodes_list:
        matrix[node1['node']-1][node2['node']-1] = vals[get_type(node1, node2)]

# правая часть
f = np.array(
    [h ** 2]*(NODES*NODES)
)

# решение слау встроенным решателем
# sol = np.linalg.solve(matrix, f)

# решение слау методом зейделя
sol = seidel(
    matrix,
    f,
    1E-5,
    norm_avg_square
)
# X = []
# Y = []
# Z = []

# for node in nodes_list:
#     X.append(node['i'] * h)
#     Y.append(node['j'] * h)

# with open('res.txt', 'w+') as f:
#     for x, y, z in zip(X, Y, sol):
#         f.write('{0:.7f} {0:.7f} {0:.7f}\n'.format(z, y, z))


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

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(
    mappable=surf,
    shrink=0.5,
    aspect=5
)
plt.show()
