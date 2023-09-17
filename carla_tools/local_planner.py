# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import Enum
from collections import deque
import random

import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import draw_waypoints
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
import os
import skgeom as sg
from skgeom.draw import draw
from PIL import Image, ImageDraw
from skgeom import minkowski
from pytope import Polytope


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
        self.astar_time_step = 0

    def __eq__(self, other):
        return self.position == other.position

def astar_v0(maze, start, end, step_size=8):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)
    search_step = 0
    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        if search_step > 500:
            return [current_node.position, current_node.position]

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0 * step_size, -1 * step_size), (0 * step_size, 1 * step_size), (-1 * step_size, 0 * step_size),
                             (1 * step_size, 0 * step_size), (-1 * step_size, -1 * step_size), (-1 * step_size, 1 * step_size),
                             (1 * step_size, -1 * step_size), (1 * step_size, 1 * step_size)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            # print('maze[node_position[0]][node_position[1]]: ', maze[node_position[0]][node_position[1]])
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

        search_step += 1

def mask2polygon(mask, thres=0.):
    # print('mask.shape: ', mask.shape)  # mask.shape:  (128, 128)
    contours, _ = cv2.findContours((mask>thres).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('contours: ', len(contours))
    # polygons = contours
    img = cv2.drawContours(np.zeros((mask.shape[0], mask.shape[1], 3)), contours, -1, (0, 0, 255), thickness=-1)
    img_mask = (img[:, :, 2] > 0).astype(np.float32)
    # print('img_mask.max(): {}, img_mask.min(): {}'.format(img_mask.max(), img_mask.min()))
    cv2.imwrite(os.path.join('{}_{:06d}.png'.format('contours', 1)), img)
    polygons = []

    for object in contours:
        if len(object) >= 4:
            # print('len(object): ', len(object))
            coords = []

            for point in object:
                # print('point: ', point)
                coords.append([int(point[0][0]), int(point[0][1])])

            polygons.append(coords)
    return polygons, img_mask

def minkowski_sum(polygons, img, vis=False, idx=0, save_img_dir='save_img_dir_minkowski'):
    os.makedirs(save_img_dir, exist_ok=True)
    # print('polygons: ', polygons)
    polygons_ego_car = polygons[0]
    print('polygons_ego_car: ', polygons_ego_car)
    # https://scikit-geometry.github.io/scikit-geometry/polygon.html#Minkowski-Sum-of-2-Polygons
    all_results = []

    V1 = np.array(polygons_ego_car)
    print(V1.shape)
    P1 = Polytope(V1)
    # Plot the Minkowski sum of two squares
    fig2, ax2 = plt.subplots()
    ax2.xaxis.tick_top()  # move x-axis to top
    ax2.grid()
    ax2.axis([0, img.shape[0], 0, img.shape[1]])
    ax2.invert_yaxis()
    P1.plot(ax2, fill=False, edgecolor=(1, 0, 0))
    for i, each_polygon in enumerate(polygons):
        if i == 0:
            continue
        V2 = np.array(polygons_ego_car)
        P2 = Polytope(V2)
        result = P1 + P2

        P2.plot(ax2, fill=False, edgecolor=(0, 0, 1))
        result.plot(ax2, fill=False,
                 edgecolor=(1, 0, 1), linestyle='--', linewidth=2)
        # for p in P10_1.V:  # the smaller square + each of the vertices of the larger one
        #   (P10_2 + p).plot(ax2, facecolor='grey', alpha=0.4,
        #                    edgecolor='k', linewidth=0.5)
        if vis:
            plt.show()
        all_results.append(result)
    ax2.legend((r'$P$', r'$Q$', r'$P \oplus Q$'))
    ax2.set_title('Minkowski sum of two polytopes')
    plt.setp([ax2], xlabel=r'$x_1$', ylabel=r'$x_2$')
    plt.savefig(os.path.join(save_img_dir, 'MinkowskiSum_{:06d}.png'.format(idx)))

    img = Image.new('L', (img.shape[0], img.shape[1]), 0)
    for each_polygon in all_results:
        ImageDraw.Draw(img).polygon(each_polygon.V, outline=1, fill=1)
    minkowski_sum_mask = np.array(img)
    return minkowski_sum_mask

def astar(maze, start, end, step_size=3, time_step_thres=5, num_search_step=1000, ego_car_size=[1, 1]):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    """
    time_step_thres: 0-9
    Cost-aware A* Alg
    """
    ego_car_size = [step_size, step_size]
    # print('---------------------------------------- ego_car_size: ', ego_car_size)
    # Create start and end node
    maze_max_value = maze.max()
    # print('maze_max_value: ', maze_max_value, 'maze.min(): ', maze.min())  # 10 maze_max_value:  20 maze.min():  0
    maze = maze_max_value - maze
    maze[start[0] + 5:] = 255
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = start_node.astar_time_step = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = end_node.astar_time_step = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)
    search_step = 0
    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # print('search_step: {}, current_node: {}, end_node: {}'.format(search_step, current_node.position, end_node.position))
        if search_step > num_search_step:
            return [current_node.position, current_node.position]

        # Found the goal
        # print('*********************************', current_node.position, end_node.position, abs(current_node.position[0] - end_node.position[0]), abs(current_node.position[1] - end_node.position[1]))
        if current_node == end_node or ((abs(current_node.position[0] - end_node.position[0]) < step_size) and (abs(current_node.position[1] - end_node.position[1]) < step_size)):
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0 * step_size, -1 * step_size), (0 * step_size, 1 * step_size), (-1 * step_size, 0 * step_size),
                             (1 * step_size, 0 * step_size), (-1 * step_size, -1 * step_size), (-1 * step_size, 1 * step_size),
                             (1 * step_size, -1 * step_size), (1 * step_size, 1 * step_size)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            # print('node_position: {}'.format(node_position))
            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                # print('node_position[0] > (len(maze) - 1): {}, node_position[0]: {}, node_position[1] > (len(maze[len(maze)-1]) -1): {}, node_position[1]: {}'
                #       .format(node_position[0] > (len(maze) - 1), node_position[0], node_position[1] > (len(maze[len(maze)-1]) -1), node_position[1]))
                continue

            # Make sure walkable terrain
            # print('maze[node_position[0]][node_position[1]]: ', maze[node_position[0]][node_position[1]])
            # if maze[node_position[0]][node_position[1]] > time_step_thres:
            #     continue
            # print(np.max(maze[(node_position[0]-ego_car_size[0]):(node_position[0]+ego_car_size[0]), (node_position[1]-ego_car_size[1]):(node_position[1]+ego_car_size[1])]),
            #       np.min(maze[(node_position[0]-ego_car_size[0]):(node_position[0]+ego_car_size[0]), (node_position[1]-ego_car_size[1]):(node_position[1]+ego_car_size[1])]),
            #       (maze[(node_position[0]-ego_car_size[0]):(node_position[0]+ego_car_size[0]),(node_position[1]-ego_car_size[1]):(node_position[1]+ego_car_size[1])] > time_step_thres).any())
            # 10, 0
            # plt.imshow(maze * 20, cmap='gray')
            # plt.show()
            # if (maze[(node_position[0]-ego_car_size[0]):(node_position[0]+ego_car_size[0]),(node_position[1]-ego_car_size[1]):(node_position[1]+ego_car_size[1])] > time_step_thres).any():
            if (maze[(node_position[0] - ego_car_size[0]):(node_position[0] + ego_car_size[0]),
                    (node_position[1] - ego_car_size[1]):(node_position[1] + ego_car_size[1])] > current_node.astar_time_step).any():
                # print('maze[(node_position[0]-ego_car_size[0]):(node_position[0]+ego_car_size[0]),(node_position[1]-ego_car_size[1]):(node_position[1]+ego_car_size[1])]).max: {}'.format((maze[(node_position[0]-ego_car_size[0]):(node_position[0]+ego_car_size[0]),(node_position[1]-ego_car_size[1]):(node_position[1]+ego_car_size[1])]).max))
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h
            child.astar_time_step = current_node.astar_time_step + 1

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

        search_step += 1

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class ImageBasedLocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None, planner_input_modality='frame', img_step_thres=[100], args=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        self._goal_location = None
        self._fake_goal_location = None
        self._goal_reached = False
        self._fake_goal_reached = False
        self._receding_horizon = 0
        self.ego_vehicle_location_offset=[75 + 128 + 500, 866]
        self.step_index = 0
        self.goal_distance = [0, 78]  # [0, 40]  # [-20, 40]
        self.num_path_point = 10 # 2
        self.save_img_dir = 'save_img_dir'
        self.planner_input_modality = planner_input_modality
        self.time_step_thres = 9
        self.img_step_thres = img_step_thres
        self.args = args
        self.path = None
        if self.planner_input_modality == 'polygon':
            self.time_step_thres = 0
        if self.save_img_dir is not None:
            os.makedirs(self.save_img_dir, exist_ok=True)
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
            print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        self._max_brake = 0.3
        self._max_throt = 0.75
        self._max_steer = 0.8
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.2,
            'K_I': 0.07,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': self._dt}
        self._offset = 0

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                                        opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']
            if 'max_throttle' in opt_dict:
                self._max_throt = opt_dict['max_throttle']
            if 'max_brake' in opt_dict:
                self._max_brake = opt_dict['max_brake']
            if 'max_steering' in opt_dict:
                self._max_steer = opt_dict['max_steering']
            if 'offset' in opt_dict:
                self._offset = opt_dict['offset']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(self, current_plan):
        """
        Resets the waypoint queue and buffer to match the new plan. Also
        sets the global_plan flag to avoid creating more waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :return:
        """

        # Reset the queue
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW

        # and the buffer
        self._waypoint_buffer.clear()
        for _ in range(self._buffer_size):
            if self._waypoints_queue:
                self._waypoint_buffer.append(
                    self._waypoints_queue.popleft())
            else:
                break

        self._global_plan = True

    def run_step(self, grid_img=None, debug=False, grid_img_wo_ego=None, ego_car_size=[1, 1], img_step=0):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """
        if self.args is not None and self.args.num_view == 2:
            if img_step < self.img_step_thres[0]:
                self.ego_vehicle_location_offset = [75 + 128 + 500, 866 - 00]
            else:
                self.ego_vehicle_location_offset = [75 + 128 + 500, 866 + 128 - 00]
        elif self.args is not None and self.args.num_view == 4:
            # # camera height 13
            # if img_step < self.img_step_thres[0]:
            #     self.ego_vehicle_location_offset = [75 + 128 + 500, 866 - 00]
            # elif img_step < self.img_step_thres[1] and img_step >= self.img_step_thres[0]:
            #     self.ego_vehicle_location_offset = [75 + 128 + 500, 866 + 128 - 00]
            #     # self.goal_distance = [78, 0]
            # elif img_step < self.img_step_thres[2] and img_step >= self.img_step_thres[1]:
            #     self.ego_vehicle_location_offset = [75 + 128 + 500, 866 + 128 - 00]
            #     self.goal_distance = [78, 0]
            # elif img_step < self.img_step_thres[3] and img_step >= self.img_step_thres[2]:
            #     self.ego_vehicle_location_offset = [75 + 128 + 500 + 210, 866 + 128 - 00]
            #     self.goal_distance = [78, 0]
            # elif self.img_step_thres[4] > img_step >= self.img_step_thres[3]:
            #     self.ego_vehicle_location_offset = [75 + 128 + 500 + 210, 866 + 128 - 00]
            #     self.goal_distance = [0, -78]
            # elif self.img_step_thres[5] > img_step >= self.img_step_thres[4]:
            #     self.ego_vehicle_location_offset = [75 + 128 + 500 + 210, 866 + 128 - 00 - 188]
            #     self.goal_distance = [0, -78]
            # camera height 15
            # [vertical (higher -> lower), horizontal (higher --> lefter)]
            delta_view4_to_view2_view12 = 7
            delta_view4_to_view2_view34 = -2
            delta_view4_to_view2_view3_right = 20
            if img_step < self.img_step_thres[0]:
                self.ego_vehicle_location_offset = [75 + 128 + 500 - delta_view4_to_view2_view12, 866 - 00 - 20]
                # self.goal_distance = [0, 78]
            elif img_step < self.img_step_thres[1] and img_step >= self.img_step_thres[0]:
                self.ego_vehicle_location_offset = [75 + 128 + 500 - delta_view4_to_view2_view12, 866 + 128 - 20 - 5 - 20]
                # self.goal_distance = [0, 78]
            elif img_step < self.img_step_thres[2] and img_step >= self.img_step_thres[1]:
                self.ego_vehicle_location_offset = [75 + 128 + 500 - delta_view4_to_view2_view12 - 10, 866 + 128 - 20]
                self.goal_distance = [-78, 0]
            elif img_step < self.img_step_thres[3] and img_step >= self.img_step_thres[2]:
                self.ego_vehicle_location_offset = [75 + 128 + 500 + 210 - delta_view4_to_view2_view34-5, 866 + 128 - 00 + 5]
                self.goal_distance = [-78, 0]
            elif self.img_step_thres[4] > img_step >= self.img_step_thres[3]:
                self.ego_vehicle_location_offset = [75 + 128 + 500 + 210 - delta_view4_to_view2_view34, 866 + 128 - 00 + delta_view4_to_view2_view3_right - 5]  #5
                self.goal_distance = [0, -78]
            elif self.img_step_thres[5] > img_step >= self.img_step_thres[4]:
                self.ego_vehicle_location_offset = [75 + 128 + 500 + 210 - delta_view4_to_view2_view34, 866 + 128 - 00 - 188 + delta_view4_to_view2_view3_right - 25 - 15]
                self.goal_distance = [0, -78]
        else:
            raise NotImplementedError

        # not enough waypoints in the horizon? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for _ in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        vehicle_transform = self._vehicle.get_transform()
        self._current_waypoint = self._map.get_waypoint(vehicle_transform.location)
        print('0:planner self._current_waypoint: ', self._current_waypoint)
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        print('1:planner self.target_waypoint: ', self.target_waypoint)
        # self.target_waypoint:  Waypoint(Transform(Location(x=26.658632, y=302.544983, z=0.000000), Rotation(pitch=360.000000, yaw=179.960953, roll=0.000000)))
        # start = (int(self._current_waypoint.transform.location.x / 0.22 - self.ego_vehicle_location_offset[0]),
        #          int(self._current_waypoint.transform.location.y / 0.22 - self.ego_vehicle_location_offset[1]))
        start = (np.clip(int(self.ego_vehicle_location_offset[0] - self._current_waypoint.transform.location.x / 0.22), 0, grid_img_wo_ego.shape[0]),
                 np.clip(int(self._current_waypoint.transform.location.y / 0.22 - self.ego_vehicle_location_offset[1]), 0, grid_img_wo_ego.shape[1]))
        end = (np.clip(int(start[0] + self.goal_distance[0]), 0, grid_img_wo_ego.shape[0]),
               np.clip(int(start[1] + self.goal_distance[1]), 0, grid_img_wo_ego.shape[1]))
        plot_path = True
        if (not 0 <= int(self.ego_vehicle_location_offset[0] - self._current_waypoint.transform.location.x / 0.22) < grid_img_wo_ego.shape[0]) \
            or (not 0 <= int(self._current_waypoint.transform.location.y / 0.22 - self.ego_vehicle_location_offset[1]) < grid_img_wo_ego.shape[1]) \
            or (not 0 <= int(start[0] + self.goal_distance[0]) < grid_img_wo_ego.shape[0]) \
            or (not 0 <= int(start[1] + self.goal_distance[1]) < grid_img_wo_ego.shape[1]):
            plot_path = False
        print('plot_path: ', plot_path)
        if self.args.num_view == 2:
            if self._goal_location is None and img_step > self.img_step_thres[0]:
                self._goal_location = end
            if img_step > self.img_step_thres[0]:
                if ((start[0] - self._goal_location[0]) ** 2 + (start[1] - self._goal_location[1]) ** 2) ** 0.5 < 5:
                    self._goal_reached = True
                else:
                    print('-----------------------',
                          ((start[0] - self._goal_location[0]) ** 2 + (start[1] - self._goal_location[1]) ** 2) ** 0.5)
        elif self.args.num_view == 4:
            if self._goal_location is None and img_step > self.img_step_thres[4]:
                self._goal_location = end
            if img_step > self.img_step_thres[4]:
                if ((start[0] - self._goal_location[0])**2 + (start[1] - self._goal_location[1])**2)**0.5 < 5:
                    self._goal_reached = True
                else:
                    print('-----------------------', ((start[0] - self._goal_location[0])**2 + (start[1] - self._goal_location[1])**2)**0.5)
        if self.args.baseline_type in ['no_replan']:
            if self.args.num_view == 2:
                if self._fake_goal_location is None and img_step > 30 and img_step < self.img_step_thres[0]:
                    self._fake_goal_location = end
            elif self.args.num_view == 4:
                if self._fake_goal_location is None and img_step > 30 and img_step < self.img_step_thres[0]:
                    self._fake_goal_location = end
            # if img_step < self.img_step_thres:
            if self._fake_goal_location is not None:
                if ((start[0] - self._fake_goal_location[0])**2 + (start[1] - self._fake_goal_location[1])**2)**0.5 < 40:
                    self._fake_goal_reached = True
                else:
                    print('------',
                          ((start[0] - self._fake_goal_location[0]) ** 2 + (start[1] - self._fake_goal_location[1]) ** 2) ** 0.5)
        print('planner start: {}, end: {}'.format(start, end))
        if self.planner_input_modality == 'polygon':
            # print('-------------------- 1. grid_img_wo_ego.shape', grid_img_wo_ego.shape)  # (128, 128)
            polygons, grid_img_wo_ego = mask2polygon(grid_img_wo_ego, thres=0.)
            # print('-------------------- 2. grid_img_wo_ego.shape', grid_img_wo_ego.shape)  # (128, 128)
        # grid_img_after_minkowski = minkowski_sum(polygons, grid_img, idx=self.step_index)
        # grid_img[:int(start[0]) - 10] = 1
        # grid_img[int(start[0]) + 10:] = 1
        image = copy.deepcopy(grid_img_wo_ego)
        if plot_path:
            # if 2 < start[0] < image.shape[0] - 2 and 2 < start[1] < image.shape[1] - 2:
            image[np.clip(start[0]-2, 0, image.shape[0]):np.clip(start[0]+2, 0, image.shape[1]),
                  np.clip(start[1]-2, 0, image.shape[0]):np.clip(start[1]+2, 0, image.shape[1])] = 0.4 * image.max()
            # if 2 < end[0] < image.shape[0]-2 and 2 < end[1] < image.shape[1]-2:
            image[np.clip(end[0] - 2, 0, image.shape[0]):np.clip(end[0] + 2, 0, image.shape[1]),
                  np.clip(end[1] - 2, 0, image.shape[1]):np.clip(end[1] + 2, 0, image.shape[1])] = 0.6 * image.max()
            image[np.clip(end[0] - 1, 0, image.shape[0]):np.clip(end[0] + 1, 0, image.shape[1]),
                  np.clip(end[1] - 1, 0, image.shape[1]):np.clip(end[1] + 1, 0, image.shape[1])] = image.max()
        # image = cv2.circle(image, (end[1], end[0]), radius=1, color=(0, 0, 255), thickness=-1)
        path = astar(grid_img_wo_ego, start=start, end=end, ego_car_size=ego_car_size, time_step_thres=self.time_step_thres)
        print(self.args.baseline_type, self.path)
        if self.args.baseline_type in ['no_replan']:
            if self.path is None:
                # print('img_step: ', img_step)
                # raise NotImplementedError
                self.path = path
            else:
                path = self.path[0:1]
                path.extend(self.path[min(img_step, len(self.path)-1):])
                # print('xxxxxxxxxxx', img_step, 'self.path: ', self.path, 'path: ', path, self.path[0:1], self.path[min(img_step, len(self.path)-1):],
                #       self.path[0:1].extend(self.path[min(img_step, len(self.path)-1):]))

        target_location = self.target_waypoint.transform.location
        # print('*************************** 1 target location: ', target_location, 'path: ', path, 'self.path: ', self.path)
        if path is not None and len(path) > 1:
            # path = path[::int(len(path) / min(self.num_path_point, len(path)))]
            print('path: ', path)
            self._receding_horizon = len(path)
            target_location.x = (self.ego_vehicle_location_offset[0] - path[1][0]) * 0.22
            target_location.y = (path[1][1] + self.ego_vehicle_location_offset[1]) * 0.22
            print('2:planner target location: ', target_location)
            self.target_waypoint = self._map.get_waypoint(target_location)
        else:
            if self.args.baseline_type in ['nocomm']:
                self.target_waypoint = self._current_waypoint
            self._receding_horizon = 0
        # self.target_waypoint = self._current_waypoint.get_left_lane()
        # target_waypoint_location = (int(self.target_waypoint.transform.location.x / 0.22 - self.ego_vehicle_location_offset[0]),
        #          int(self.target_waypoint.transform.location.y / 0.22 - self.ego_vehicle_location_offset[1]))
        target_waypoint_location = (
        int(self.ego_vehicle_location_offset[0] - self.target_waypoint.transform.location.x / 0.22),
        int(self.target_waypoint.transform.location.y / 0.22 - self.ego_vehicle_location_offset[1]))
        print('3:planner target_waypoint_location: ', target_waypoint_location)
        # if 3 < target_waypoint_location[0] < image.shape[0] - 3 and 3 < target_waypoint_location[1] < image.shape[1] - 3:
        if plot_path:
            image[max(0, target_waypoint_location[0] - 3):min(image.shape[0], target_waypoint_location[0] + 3),
            max(0, target_waypoint_location[1] - 3):min(image.shape[1], target_waypoint_location[1] + 3)] = 0.6 * image.max()
        print('4:planner self.target_waypoint', self.target_waypoint)
        # move using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)
        # print('control: ', control)
        if self.save_img_dir is not None:
            # print('\n\n\n\n\nimage.max(), image.min(): ', image.max(), image.min())
            if path is not None:
                for each_point in path:
                    # print('each_point', each_point)
                    each_x = int(each_point[0])  # int((each_point[0] + self.ego_vehicle_location_offset[0]) * 0.22)
                    each_y = int(each_point[1])  # int((each_point[1] + self.ego_vehicle_location_offset[0]) * 0.22)
                    # print('each_x: {}, each_y: {}'.format(each_x, each_y))
                    distance = 1
                    image[max(0, each_x - distance):min(image.shape[0], each_x + distance),
                          max(0, each_y - distance):min(image.shape[1], each_y + distance)] = 0.5 * image.max()
            # plt.imshow(image, cmap='gray',)
            # plt.show()
            cv2.imwrite(os.path.join(self.save_img_dir, 'grid_img_{0:06d}.png'.format(img_step)),
                        image.astype(np.float32) * 255. / image.max())
        # purge the queue of obsolete waypoints
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if waypoint.transform.location.distance(vehicle_transform.location) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        # if debug:
        #     draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)
        self.step_index += 1
        return control

    def done(self):
        """
        Returns whether or not the planner has finished

        :return: boolean
        """
        return len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
            print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        self._max_brake = 0.3
        self._max_throt = 0.75
        self._max_steer = 0.8
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.2,
            'K_I': 0.07,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': self._dt}
        self._offset = 0

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                                        opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']
            if 'max_throttle' in opt_dict:
                self._max_throt = opt_dict['max_throttle']
            if 'max_brake' in opt_dict:
                self._max_brake = opt_dict['max_brake']
            if 'max_steering' in opt_dict:
                self._max_steer = opt_dict['max_steering']
            if 'offset' in opt_dict:
                self._offset = opt_dict['offset']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(self, current_plan):
        """
        Resets the waypoint queue and buffer to match the new plan. Also
        sets the global_plan flag to avoid creating more waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :return:
        """

        # Reset the queue
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW

        # and the buffer
        self._waypoint_buffer.clear()
        for _ in range(self._buffer_size):
            if self._waypoints_queue:
                self._waypoint_buffer.append(
                    self._waypoints_queue.popleft())
            else:
                break

        self._global_plan = True

    def run_step(self, debug=False):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """

        # not enough waypoints in the horizon? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for _ in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        vehicle_transform = self._vehicle.get_transform()
        self._current_waypoint = self._map.get_waypoint(vehicle_transform.location)
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        # purge the queue of obsolete waypoints
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if waypoint.transform.location.distance(vehicle_transform.location) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        # if debug:
        #     draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)

        return control

    def done(self):
        """
        Returns whether or not the planner has finished

        :return: boolean
        """
        return len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0

def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
