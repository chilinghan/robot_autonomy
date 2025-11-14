#!/usr/bin/env python3

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.math_utils import wrap_angle

import rclpy
import numpy as np
from scipy.interpolate import splev, splrep

class Navigator(BaseNavigator):
    def __init__(self, kw=2.0, kpx=2.0, kpy=2.0,
                 kdx=2.0, kdy=2.0):
        super().__init__("navigator")
        self.kw = kw
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        self.V_PREV_THRES = 0.0001

    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def compute_heading_control(self,
                                state: TurtleBotState,
                                goal: TurtleBotState) -> TurtleBotControl:
        error = wrap_angle(state.theta - goal.theta)
        return TurtleBotControl(omega=-self.kw * error)
        

    def compute_trajectory_tracking_control(self, state, plan, t):
        dt = t - self.t_prev

        x_d, y_d = plan.desired_state(t).x, plan.desired_state(t).y
        xd_d, yd_d = splev(t, plan.path_x_spline, der=1), splev(t, plan.path_y_spline, der=1)
        xdd_d, ydd_d = splev(t, plan.path_x_spline, der=2), splev(t, plan.path_x_spline, der=2)

        xd = self.V_prev * np.cos(state.theta)
        yd = self.V_prev * np.sin(state.theta)
        u1 = xdd_d + self.kpx*(x_d - state.x) + self.kdx*(xd_d - xd)
        u2 = ydd_d + self.kpy*(y_d - state.y) + self.kdy*(yd_d - yd)
        
        vd = u1*np.cos(state.theta) + u2*np.sin(state.theta)
        V = max(self.V_prev + vd*dt, self.V_PREV_THRES)
        om = (-u1*np.sin(state.theta) + u2*np.cos(state.theta))/V

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return TurtleBotControl(v=V, omega=om)

    
    def compute_trajectory_plan(self, state, goal, occupancy, resolution, horizon):
        astar = AStar((0, 0), (horizon, horizon), state, goal, occupancy, resolution)

        if not astar.solve() or len(astar.path) < 4:
            return None
        self.reset()

        v_desired, spline_alpha = 0.15, 0.05
        path = np.asarray(astar.path)
        ts = [0]
        for i in range(1, len(path)):
            ts.append(ts[-1] + np.linalg.norm((path[i] - path[i-1]))/v_desired)

        ts = np.asarray(ts)
        path_x_spline = splrep(ts, path[:, 0], s=spline_alpha)
        path_y_spline = splrep(ts, path[:, 1], s=spline_alpha)
        
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = (x_init.x, x_init.y)                
        self.x_init = self.snap_to_grid((x_init.x, x_init.y))    # initial state
        self.x_goal = self.snap_to_grid((x_goal.x, x_goal.y))    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        in_bounds = min(x) > max(self.statespace_lo) and max(x) < min(self.statespace_hi)
        return in_bounds and self.occupancy.is_free(np.array(x))
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for i in [-self.resolution, 0, self.resolution]:
            for j in [-self.resolution, 0, self.resolution]:
                if i == 0 and j == 0:
                    continue
                neighbor = (x[0] + i, x[1] + j)
                if self.is_free(neighbor):
                    neighbors.append(self.snap_to_grid(neighbor))
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        self.open_set.add(self.x_init)
        while self.open_set:
            x = self.find_best_est_cost_through()

            if x == self.x_goal:
                self.path = self.reconstruct_path()
                return True

            self.open_set.remove(x)
            self.closed_set.add(x)
            for neighbor in self.get_neighbors(x):
                if neighbor not in self.closed_set:
                    new_cost = self.cost_to_arrive[x] + self.distance(neighbor, x)
                    
                    if new_cost < self.cost_to_arrive.get(neighbor, float("inf")):
                        if neighbor not in self.open_set:
                            self.open_set.add(neighbor)
                        self.est_cost_through[neighbor] = new_cost + self.distance(neighbor, self.x_goal)
                        self.cost_to_arrive[neighbor] = new_cost
                        self.came_from[neighbor] = x

        return False
        ########## Code ends here ##########


if __name__ == "__main__":
    rclpy.init()
    navigatorNode = Navigator()
    rclpy.spin(navigatorNode)
    rclpy.shutdown()
