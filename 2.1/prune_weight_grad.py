import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)



X, Y = np.meshgrid(x, y)

Z = -1 * np.sin(np.sqrt(X**2 + Y ** 2)) / (np.sqrt(X**2+ Y**2))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_xlabel('y')
ax.set_zlabel('loss')
plt.show()
