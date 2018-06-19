# author - Mayur Bency
# last edited - 11/13/2017

from __future__ import print_function
import heapq
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
import sys


class GridWorld:
    """Creates a 2D grid with given specs."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacle = []

    def boundary(self, pos):
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def legal_move(self, pos):
        return pos not in self.obstacle

    def neighbors(self, pos):
        (x, y) = pos
        next_move = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        next_move = filter(self.boundary, next_move)
        next_move = filter(self.legal_move, next_move)
        return next_move

    def cost(self, from_node, to_node):
        return 1


class PriorityQueue:
    """Initializes an empty queue ordered according to priority."""
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def heuristic(a, b):
    """Returns the estimated heuristic h(n) """
    (ax, ay) = a
    (bx, by) = b
    return abs(bx - ax) + abs(by - ay)  # returns the manhattan distance


def search(g, robot_pose, goal_pose, max_step_number, planner, verbose):
    """Returns a path given query points, world, and type of planner selected"""
    if planner == 'random':
        path = [robot_pose]
        current_step = robot_pose
        step_number = 0
        # create and initialize memory bank
        q = collections.deque([robot_pose], maxlen=int(np.sqrt(max_step_number)))
        memory_filter = lambda x: x not in q
        while True:
            potential_moves = list(g.neighbors(current_step))
            filtered_potential_moves = list(filter(memory_filter, potential_moves))
            if not filtered_potential_moves:
                next_move = potential_moves[random.sample(range(len(potential_moves)), 1)[0]]
            else:
                next_move = filtered_potential_moves[random.sample(range(len(filtered_potential_moves)), 1)[0]]
            q.append(next_move)
            path.append(next_move)
            current_step = next_move
            step_number += 1
            if current_step == goal_pose:
                if verbose == 1:
                    print('Feasible path found! Exiting...')
                    print('Steps taken: ', step_number)
                return np.asarray(path), step_number

            if step_number > max_step_number:
                if verbose == 1:
                    print('Search failed. Try increasing max_step_number')
                return np.asarray(path), step_number

    elif planner == 'optimal':
        # create a queue of explored nodes ranked by heuristic value
        explored = PriorityQueue()
        explored.put(robot_pose, 0)
        came_from = {}
        cost_so_far = {}
        came_from[robot_pose] = None
        cost_so_far[robot_pose] = 0
        counter = 0

        while not explored.empty():
            current_step = explored.get()

            if current_step == goal_pose:
                if verbose == 1:
                    print('Optimal path found! Exiting...')
                    print('Nodes explored: ', counter)
                break

            for next_move in g.neighbors(current_step):
                new_cost = cost_so_far[current_step] + g.cost(current_step, next_move)
                if next_move not in cost_so_far or new_cost < cost_so_far[next_move]:
                    cost_so_far[next_move] = new_cost
                    priority = new_cost + heuristic(goal_pose, next_move)
                    explored.put(next_move, priority)
                    came_from[next_move] = current_step
                    counter += 1
        current_step = goal_pose
        path = [current_step]
        while current_step != robot_pose:
            current_step = came_from[current_step]
            path.append(current_step)
        path.append(robot_pose)
        path.reverse()
        return np.asarray(path), counter

    else:
        print('Invalid planner selection. Try again!')


def visualize_planner(g, robot_pose, goal_pose, path):
    """Plot the world, generated path, and query points after you're done"""
    plt.close('all')
    plt.figure()
    obstacles = np.asarray(g.obstacle)
    plt.plot(obstacles[:, 0], obstacles[:, 1], 'ms', markersize=32)
    plt.xlim([-1, g.width])
    plt.ylim([-1, g.height])
    plt.plot(path[:, 0], path[:, 1], 'k.-', markersize=20)
    plt.plot(robot_pose[0], robot_pose[1], 'g.', markersize=20)
    plt.plot(goal_pose[0], goal_pose[1], 'r.', markersize=20)
    plt.show()


def main():
    """An example case"""
    # create and initialize a workspace
    g = GridWorld(width=10, height=10)
    g.obstacle = [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
                  (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
                  (3, 9), (3, 8), (3, 7), (3, 6), (3, 5), (3, 4),
                  (4, 9), (4, 8), (4, 7), (4, 6), (4, 5), (4, 4)]

    # set query points
    planner_selection = sys.argv[1]  # takes in value through command line

    if len(sys.argv) > 2:
        init_pose = (int(sys.argv[2]), int(sys.argv[3]))
        fin_pose = (int(sys.argv[4]), int(sys.argv[5]))
    else:
        init_pose = (1, 6)
        fin_pose = (9, 0)

    max_step_number = 200

    # select planner and generate path
    random_path, _ = search(g, init_pose, fin_pose, max_step_number, planner=planner_selection, verbose=1)

    # visualize planner output
    visualize_planner(g, init_pose, fin_pose, random_path)


if __name__ == '__main__':
    main()




