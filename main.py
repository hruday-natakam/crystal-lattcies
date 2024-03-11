import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

# Data for plots
x = np.array([1, 0.5, 0, 0, 1])
y = np.array([0, 0.5, 1, 0, 0])
a = np.array([0])
b = np.array([0])

# Function to update right plot based on hover coordinates
def update_plot(event, sigma):
    hover_x = event.xdata
    hover_y = event.ydata

    # Calculate values based on hover input
    if hover_x is not None and hover_y is not None:
        if hover_x <= 1 - hover_y:
            r12 = (sigma / 3) * hover_y
            r01 = (sigma / 6) * (3 - 3 * hover_x - hover_y)
            r02 = (sigma / 6) * (3 + 3 * hover_x - hover_y)
            v1 = np.sqrt(r12**2 + r01**2)
            v2 = np.sqrt(r12**2 + r02**2)
            ang = np.arccos((-4 * hover_y**2) / np.sqrt((9 * hover_x**2 + 5 * hover_y**2 - 6 * hover_y + 9)**2 - 36 * hover_x**2 * (3 - hover_y)**2))

            # Clear the previous plot
            right_ax.clear()
            right_ax.plot(a, b, 'ko', markersize=5)  # Original point
            right_ax.add_patch(FancyArrowPatch((a[0], b[0]), (a[0] + v1, b[0]), color='green', arrowstyle='->', mutation_scale=20))
            right_ax.add_patch(FancyArrowPatch((a[0], b[0]), (a[0] + v2 * np.cos(ang), b[0] + v2 * np.sin(ang)), color='blue', arrowstyle='->', mutation_scale=20))
            right_ax.add_patch(FancyArrowPatch((a[0], b[0]), (a[0] - v1 - v2 * np.cos(ang), b[0] - v2 * np.sin(ang)), color='red', arrowstyle='->', mutation_scale=20))
            right_ax.set_aspect('equal')
            right_ax.set_xlabel('x')
            right_ax.set_ylabel('y')
            right_ax.set_title('Its co-ordinates')
            right_ax.set_xlim(-1, 1)
            right_ax.set_ylim(-1, 1)

            plt.draw()

# Plot 1
fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(10, 5))

polygon = Polygon(np.vstack((x, y)).T, color='yellow')
left_ax.add_patch(polygon)
left_ax.set_aspect('equal')
left_ax.set_xlabel('x')
left_ax.set_ylabel('y')
left_ax.set_title('Lattices')

# Define sigma
sigma = 1

# Connect hover event to update function
fig.canvas.mpl_connect('motion_notify_event', lambda event: update_plot(event, sigma))

plt.tight_layout()
plt.show()
