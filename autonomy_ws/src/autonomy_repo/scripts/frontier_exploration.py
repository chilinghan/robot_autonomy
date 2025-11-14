#!/usr/bin/env python3

from rclpy.node import Node
from scipy.signal import convolve2d
import numpy as np
from std_msgs.msg import Bool
from asl_tb3_lib.grids import StochOccupancyGrid2D
from nav_msgs.msg import OccupancyGrid
from asl_tb3_msgs.msg import TurtleBotState

class FrontierExploration(Node):
    def __init__(self):
        self.__init__("frontier_exploration")
        self.create_subscription(Bool, "/nav_success", self.success_callback)
        self.create_subscription(OccupancyGrid, "/map", self.map_callback)
        self.create_subscription(TurtleBotState, "/state", self.state_callback)
        self.nav_command = self.create_publisher(TurtleBotState, "/cmd_nav", self.get_target_state)

        self.nav_success = True

    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

    def success_callback(self, msg: Bool) -> None:
        self.nav_success = msg.data
    
    def state_callback(self, msg: TurtleBotState) -> None:
        self.current_state = np.array([msg.x, msg.y])

    def get_target_state(self) -> None:
        if self.current_state is None:
            return
        frontier_states = self.explore()
        distances = np.linalg.norm(frontier_states - self.current_state, axis=1)
        target_state = frontier_states[np.argmin(distances)]

        state = TurtleBotState(x=target_state[0], y=target_state[1])
        if self.nav_success:
            self.nav_command.publish(state)

    def explore(self):
        if self.occupancy is None:
            return
    
        window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics

        unknown = self.occupancy.probs == -1
        occupied = self.occupancy.probs >= 0.5
        unoccupied = (self.occupancy.probs < 0.5) & (self.occupancy.probs >= 0)

        count = np.ones((window_size, window_size))
        count_unknown = convolve2d(unknown, count, mode="same")
        count_occupied = convolve2d(occupied, count, mode="same")
        count_unoccupied = convolve2d(unoccupied, count, mode="same")

        window_area = window_size**2
        cond_unknown = (count_unknown / window_area) >= 0.20
        cond_no_occupied = count_occupied == 0
        cond_unoccupied = (count_unoccupied / window_area) >= 0.30

        frontier_mask = cond_unknown & cond_no_occupied & cond_unoccupied
        frontier_idxs = np.argwhere(frontier_mask)

        frontier_grid_xy = np.column_stack([frontier_idxs[:, 1], frontier_idxs[:, 0]])

        frontier_states = np.array([self.occupancy.grid2state(grid_xy) for grid_xy in frontier_grid_xy])

        return frontier_states
