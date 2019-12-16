import matplotlib.pyplot as plt
import numpy as np

h = 0.2

X, Y = np.meshgrid(np.arange(0, 1, h), np.arange(0, 1, h))

x_shape = X.shape

U = np.zeros(x_shape[0])
V = np.zeros(x_shape[0])

for i in range(x_shape[0]):
    U[i] = 0.1
    V[i] = -0.2

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V, units='xy', scale=2, color='red')

ax.set_aspect('equal')

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.title('How to plot a vector field using matplotlib ?', fontsize=10)

# plt.savefig('how_to_plot_a_vector_field_in_matplotlib_fig1.png',
# bbox_inches='tight')
plt.show()
plt.close()
