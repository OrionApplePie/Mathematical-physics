import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags

from linalg import (
    seidel,
    seidel1,
    norm1,
    norm_euclid,
    norm_avg_square
)


def insert_boundary_values(u):
    # add values in 0 and 1, просто вствляем известные значения в крайних узлаах
    n = len(u)
    u = np.insert(
        u,
        0,
        [[0]],
        axis=0
    )
    u = np.insert(
        u,
        n + 1,
        [[0]],
        axis=0
    )
    return u


if __name__ == "__main__":
    
    # domain
    a = 0.0
    b = 1.0
    eps = 1E-5
    # h - step, n - nodes
    # print('Enter nodes number: ')
    h = 0.01
    n = 99 # inner nodes
     
    # coefficient of stiffness matrix
    a_left = a_right = (h / 6.) - (1. / h)
    a_center = (2. / h) + (2.0*h) / 3.0

    # print('n= {}, h={}, a_left={}'.format(n, h, a_left))
    
    # generating of stiffnes matrix
    A = diags(
        diagonals=[a_left, a_center, a_right],
        offsets=[-1, 0, 1],
        shape=(n, n)
    ).toarray()
    
    b = np.full(
        shape=(n, 1),
        fill_value=h
    )
    # print(A)
    # print(b)

    u_k_np = np.linalg.solve(A, b)
    
    u_k = seidel(
        A,
        b,
        eps,
        norm1
    )

    u_k2 = seidel(
        A,
        b,
        eps,
        norm_euclid
    )
    
    # with angsquare morm
    # u_k3 = seidel(
    #     A,
    #     b,
    #     eps,
    #     norm_avg_square
    # )
    # print(np.transpose(u_k).tolist())    

    u_k_np = insert_boundary_values(u_k_np)
    u_k = insert_boundary_values(u_k) 
    u_k2 = insert_boundary_values(u_k2)
    # u_k3 = insert_boundary_values(u_k3)

    # Setka 0 to 1 по всем узлам
    x_h = np.array(
        [i*h for i in range(n + 2)]
    )
    
    # print(x_h)

    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(x_h, u_k_np, 'r', label='Решатель numpy')
    plt.plot(x_h, u_k, 'g', label="Норма макс")
    plt.plot(x_h, u_k2, 'b-', label='Евклидова норма')
    # plt.plot(x_h, u_k3, 'y+')
    plt.legend()
    plt.show()
    