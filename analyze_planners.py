# author - Mayur Bency
# last edited - 11/13/2017

import numpy as np
from two_discrete_planners import search, GridWorld, visualize_planner
import random
import matplotlib.pyplot as plt
import numpy_indexed as npi


def branching_factor(n, d):
    coeff = np.zeros((1, d+2))[0]
    coeff[0] = 1
    coeff[-2] = -n+1
    coeff[-1] = -n
    b_roots = np.roots(coeff)
    return [x for x in list(b_roots) if np.isreal(x) and x > 0][0]


g = GridWorld(width=10, height=10)
g.obstacle = [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
              (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
              (3, 9), (3, 8), (3, 7), (3, 6), (3, 5), (3, 4),
              (4, 9), (4, 8), (4, 7), (4, 6), (4, 5), (4, 4)]


init_pose = (1, 6)
fin_pose = (1, 0)
max_step_number = 200

test_num = 10000
complexity = np.zeros((test_num, 1))
depth = np.zeros((test_num, 1))
effective_b = np.zeros((test_num, 1))

for i in range(test_num):
    while True:
        x1 = (random.randint(0, g.width - 1))
        y1 = (random.randint(0, g.height - 1))
        x2 = (random.randint(0, g.width - 1))
        y2 = (random.randint(0, g.height - 1))
        if (x1, y1) not in g.obstacle and (x2, y2) not in g.obstacle and (x1, y1) != (x2, y2):
            break
    path, nodes_visited = search(g, (x1, y1), (x2, y2), max_step_number, planner='random', verbose=0)
    optimal_path, _ = search(g, (x1, y1), (x2, y2), max_step_number, planner='optimal', verbose=0)

    complexity[i] = nodes_visited
    depth[i] = optimal_path.shape[0] - 1
    # calculate the effective branching factor for A*
    # effective_b[i] = branching_factor(nodes_visited, path.shape[0] - 1)

print('Mean of number of nodes visited: ', np.mean(complexity))
print('Stdev of number of nodes visited: ', np.std(complexity))
print('Mean of path length: ', np.mean(depth))
print('Stdev of path_length: ', np.std(depth))
print('Mean of effective branching: ', np.mean(effective_b))
print('Stdev of effective branching: ', np.std(effective_b))

visualize_planner(g, (1, 6), (1, 0), path)

## plot complexity versus depth of search
plt.close('all')
depth_unique, complexity_mean = npi.group_by(depth).mean(complexity)
depth_unique, complexity_stdev = npi.group_by(depth).std(complexity)
plt.plot(depth_unique, complexity_mean, label='Mean complexity')
plt.plot(depth_unique, complexity_stdev, label='Std deviation of complexity')
plt.xlabel('Depth of search')
plt.ylabel('Nodes visited (defined as complexity)')
plt.legend(frameon=False, loc='upper left')

# fit a polynomial to the complexity data
z = np.polyfit(depth_unique[:, 0], complexity_mean[:, 0], 3)
f = np.poly1d(z)
x_new = np.linspace(depth_unique[0], depth_unique[-1], 100)
y_new = f(x_new)
plt.plot(x_new, y_new, 'k--')
plt.show()
