# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

import queue
import numpy as np
import cv2, os, sys, copy, time
import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from local_planner import ImageBasedLocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
import torch
import skgeom as sg
from skgeom.draw import draw
import pdb

class BasicAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20, agent_id=0, vis=False, save_dir='save_img_dir', planner_type='ImageBased',
                 planner_input_modality='frame', args=None):
        """

        :param vehicle: actor to apply to local planner logic onto
        planner_type='ImageBased'
        """
        super(BasicAgent, self).__init__(vehicle)

        self.num_view = args.num_view
        if self.num_view == 2:
            self.img_step_thres = [100]
        elif self.num_view == 4:
            # self.img_step_thres = [80, 130, 170, 330, 450, 700]
            self.img_step_thres = [80, 130, 170, 330-15, 450, 700]
            # self.img_step_thres = [80, 130, 170, 245, 400]
            # self.img_step_thres = [80, 120, 240, 300]
            # view 1 -> view 2, turn left, view 2 -> view 3, turn left, view 3 -> view 4
        print('self.img_step_thres: ', self.img_step_thres)
        # pdb.set_trace()
        self.planner_input_modality = planner_input_modality
        self._proximity_tlight_threshold = 5.0  # meters
        self._proximity_vehicle_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self.planner_type = planner_type
        self.args = args
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.4,
            'K_I': 0,
            'dt': 1.0/20.0}
        if self.planner_type == 'ImageBased':
            self._local_planner = ImageBasedLocalPlanner(
                self._vehicle, opt_dict={'target_speed': target_speed,
                                         'lateral_control_dict': args_lateral_dict},
            planner_input_modality=planner_input_modality, img_step_thres=self.img_step_thres, args=args)
        else:
            self._local_planner = LocalPlanner(
                self._vehicle, opt_dict={'target_speed' : target_speed,
                'lateral_control_dict':args_lateral_dict})
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None
        self.traffic_light_obs = False
        self.his_length = 20
        self.q_E = queue.Queue(self.his_length)
        self.q_E_t2nd = queue.Queue(self.his_length)
        self.step_index = 0
        self.width = 50
        self.img_title = self.draw_img_title(30, 128)  # self.draw_img_title(100, 600)
        self.vis = vis
        self.first_t2no_img = None
        self.agent_id = agent_id
        self.use_pytorch_model = True # False
        self.distance_thres = 30
        self.num_view = args.num_view
        if self.num_view == 4:
            self.occupied_area_thres = 50000  # 970 # 30  # 1500 # 500
        elif self.num_view == 2:
            self.occupied_area_thres = 1100  # 970 # 30  # 1500 # 500
        self.device = 'cuda:0'
        self.collision_detection_mode = 'robot'  # 'contour'
        self.start_vis_frame_index = 10   # 150 # 65  # np.inf  # 65
        self.ego_car_size = [8, 27]  # [8, 15]
        self.forehead_distance = [5 + self.ego_car_size[0], 10 + self.ego_car_size[1]]
        self.exchange_flag = -1
        self.save_img_dir = save_dir
        self.t2no_t2nd_method = 'v2'
        self.metrics = {'NumberOfCollision': 0, 'TimeToReachGoals': [], 'TimestepsToReachGoals': 0, 'TotalControlEffort': 0,
                        'NumberOfSuddenReversalsInOrder': 0, 'GoalReached': False, 'FakeGoalReached': False,
                        'TotalRecedingHorizon': 0, 'PlanningTime': [], 'TravelDistance': 0}
        # ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred', 'results/20220605-224950/carla_town02_2_view_20220524_color_waypoint')
        ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                'results/20220829-080108_RGB/carla_town02_2_view_20220828_icra')
        if self.num_view == 2:
            input_video_name = '../tools/carla_town02_2_view_20220828_icra'
            each_view = '_out_0'
            B_0 = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
            B_0 = cv2.resize(B_0, (int(128), int(128)))
            self.B_0 = cv2.cvtColor(B_0, cv2.COLOR_BGR2GRAY)
            each_view = '_out_1'
            B_1 = cv2.imread(os.path.join(input_video_name, each_view, '00000240.png')).astype(np.uint8)
            B_1 = cv2.resize(B_1, (int(128), int(128)))
            self.B_1 = cv2.cvtColor(B_1, cv2.COLOR_BGR2GRAY)
            self.B_2 = None
            self.B_3 = None
        elif self.num_view == 4:
            input_video_name = '../tools/carla_town02_4_view_20221126_icra'
            each_view = '_out_0'
            # B_0 = cv2.imread(os.path.join(input_video_name, each_view, '00000269.png')).astype(np.uint8)
            B_0 = cv2.imread(os.path.join(input_video_name, each_view, '00000012.png')).astype(np.uint8)
            B_0 = cv2.resize(B_0, (int(128), int(128)))
            # B_0 = np.ones((128, 128, 3)).astype(np.uint8)
            # print('B_0.shape: ', B_0.shape)
            self.B_0 = cv2.cvtColor(B_0, cv2.COLOR_BGR2GRAY)
            each_view = '_out_1'
            # B_1 = cv2.imread(os.path.join(input_video_name, each_view, '00000194.png')).astype(np.uint8)
            B_1 = cv2.imread(os.path.join(input_video_name, each_view, '00000012.png')).astype(np.uint8)
            B_1 = cv2.resize(B_1, (int(128), int(128)))
            # B_1 = np.ones((128, 128, 3)).astype(np.uint8)
            self.B_1 = cv2.cvtColor(B_1, cv2.COLOR_BGR2GRAY)
            each_view = '_out_2'
            # print(os.path.join(input_video_name, each_view, '00000081.png'))
            # B_2 = cv2.imread(os.path.join(input_video_name, each_view, '00000081.png')).astype(np.uint8)
            B_2 = cv2.imread(os.path.join(input_video_name, each_view, '00000012.png')).astype(np.uint8)
            B_2 = cv2.resize(B_2, (int(128), int(128)))
            # B_2 = np.ones((128, 128, 3)).astype(np.uint8)
            self.B_2 = cv2.cvtColor(B_2, cv2.COLOR_BGR2GRAY)
            each_view = '_out_3'
            # B_3 = cv2.imread(os.path.join(input_video_name, each_view, '00000192.png')).astype(np.uint8)
            B_3 = cv2.imread(os.path.join(input_video_name, each_view, '00000012.png')).astype(np.uint8)
            B_3 = cv2.resize(B_3, (int(128), int(128)))
            # B_3 = np.ones((128, 128, 3)).astype(np.uint8)
            self.B_3 = cv2.cvtColor(B_3, cv2.COLOR_BGR2GRAY)
        if self.save_img_dir is not None:
            os.makedirs(self.save_img_dir, exist_ok=True)
        if self.use_pytorch_model:
            if self.args.baseline_type in ['monolith', 'no_replan', 'nocomm', 'allcomm']:
                if self.num_view == 2:
                    if planner_input_modality == 'frame':
                        self.memory_0 = None
                        ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                                # 'results/20220925-182538/carla_town02_2_view_20220828_icra')
                                                'results/20230430-210909/carla_town02_2_view_20220828_icra')
                    else:
                        ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                                'results/20220928-102832/carla_town02_2_view_20220828_icra')
                elif self.num_view == 4:
                    if planner_input_modality == 'frame':
                        self.memory_0 = None
                        # ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                        #                         'results/20221128-172449/carla_town02_4_view_20221126_icra')
                        ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                                'results/20230430-164404/carla_town02_4_view_20221126_icra/')
                    else:
                        if self.args.baseline_type in ['allcomm']:
                            # ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                            #                         'results/20230113-205813/carla_town02_4_view_20221126_icra')
                            ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                                    'results/20230319-150533/carla_town02_4_view_20221126_icra')
                        else:
                            # ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                            #                         'results/20221128-152109/carla_town02_4_view_20221126_icra')
                            # ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                            #                         'results/20230113-205813/carla_town02_4_view_20221126_icra')
                            if self.args.num_classes == 2:
                                ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                                        'results/20230330-124323_02/carla_town02_4_view_20221126_icra')
                            elif self.args.num_classes == 10:
                                ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                                        'results/20230319-150533/carla_town02_4_view_20221126_icra')
                            elif self.args.num_classes == 20:
                                ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                                        'results/20230319-150533/carla_town02_4_view_20221126_icra')
                            elif self.args.num_classes == 30:
                                ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                                        'results/20230330-194146_30/carla_town02_4_view_20221126_icra')
                            elif self.args.num_classes == 60:
                                ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                                        'results/20230331-002123_60/carla_town02_4_view_20221126_icra')
                model_0_path = os.path.join(ckpt_dir, "model_0.pt")
                print('Loading model from {} ...'.format(model_0_path))
                self.model_0 = torch.load(model_0_path)
                vae_path = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                        'results/', 'vae_carla', 'carla_town02_2_view_20220828_icra', 'vae.pt')
                self.vae = torch.load(vae_path)
            else:
                if planner_input_modality == 'frame':
                    self.memory_0 = None
                    self.memory_1 = None
                else:
                    # ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred', 'results/20220605-224950/carla_town02_2_view_20220524_color_waypoint')
                    # ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                    #                         'results/20220830-015349_depth/carla_town02_2_view_20220828_icra')
                    ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                            'results/20230113-205813/carla_town02_4_view_20221126_icra')
                model_0_path = os.path.join(ckpt_dir, "model_0.pt")
                # h_units = [int(x) for x in '16,16'.split(',')]
                # args = Bunch(img_width=128)
                # input_dim = 5
                # act = 'relu'
                # self.model_0 = CONV_NN_v2(input_dim, h_units, act, args)
                # self.model_0.load_state_dict(torch.load(model_0_path))
                self.model_0 = torch.load(model_0_path)
                model_1_path = os.path.join(ckpt_dir, "model_1.pt")
                self.model_1 = torch.load(model_1_path)

                vae_path = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                            'results/', 'vae_carla', 'carla_town02_2_view_20220828_icra', 'vae.pt')
                self.vae = torch.load(vae_path)
                # vae = vae.to(args.device)
        # pdb.set_trace() 

    def draw_img_title(self, height=100, width=128, color=[255, 255, 255], text_size=0.5):
        canvas = np.zeros((height, width))
        # print('canvas.shape: ', canvas.shape)  # (600, 600)  #  (30, 128)
        text_bg = "BG"
        text_frame = "Frame T"
        text_diff_t2no = "Diff T2NO"
        text_diff_t2nd = "Diff T2ND"
        text_t2no = "T2NO"
        text_t2nd = "T2ND"
        img_title_bg = cv2.putText(canvas.copy(), text_bg, (int(0.25 * self.width), int(height * 0.6)), cv2.FONT_HERSHEY_COMPLEX, text_size,
                                   color)
        img_title_frame = cv2.putText(canvas.copy(), text_frame, (int(0.25 * self.width), int(height * 0.6)), cv2.FONT_HERSHEY_COMPLEX,
                                      text_size, color)
        img_title_diff_t2no = cv2.putText(canvas.copy(), text_diff_t2no, (int(0.25 * self.width), int(height * 0.6)),
                                          cv2.FONT_HERSHEY_COMPLEX, text_size, color)
        img_title_diff_t2nd = cv2.putText(canvas.copy(), text_diff_t2nd, (int(0.25 * self.width), int(height * 0.6)),
                                          cv2.FONT_HERSHEY_COMPLEX, text_size, color)
        img_title_t2no = cv2.putText(canvas.copy(), text_t2no, (int(0.25 * self.width), int(height * 0.6)), cv2.FONT_HERSHEY_COMPLEX,
                                     text_size, color)
        img_title_t2nd = cv2.putText(canvas.copy(), text_t2nd, (int(0.25 * self.width), int(height * 0.6)), cv2.FONT_HERSHEY_COMPLEX,
                                     text_size, color)
        concat_title = np.concatenate(
            [img_title_bg, np.zeros_like(img_title_bg[:, :4]), img_title_frame, np.zeros_like(img_title_bg[:, :4]),
             img_title_diff_t2no, np.zeros_like(img_title_bg[:, :4]),
             img_title_diff_t2nd, np.zeros_like(img_title_bg[:, :4]), img_title_t2no,
             np.zeros_like(img_title_bg[:, :4]), img_title_t2nd], axis=1)
        return concat_title

    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)

        self._local_planner.set_global_plan(route_trace)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def run_step(self, debug=False, view_img=None, img_step=0):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        print('self.planner_type: ', self.planner_type)

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        planning_time_start = time.time()
        if self.planner_type == 'ImageBased':
            # check visually possible obstacles
            vehicle_state, grid_img_wo_ego, grid_img = self._is_vehicle_hazard_visual(vehicle_list, view_img, img_step)
            if vehicle_state:
                if debug:
                    print('!!! VEHICLE BLOCKING AHEAD)'.format())

                self._state = AgentState.BLOCKED_BY_VEHICLE
                hazard_detected = True
        else:
            # check possible obstacles
            vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
            if vehicle_state:
                if debug:
                    print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

                self._state = AgentState.BLOCKED_BY_VEHICLE
                hazard_detected = True

        # check for the state of the traffic lights
        if self.traffic_light_obs:
            lights_list = actor_list.filter("*traffic_light*")
            light_state, traffic_light = self._is_light_red(lights_list)
            if light_state:
                if debug:
                    print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

                self._state = AgentState.BLOCKED_RED_LIGHT
                hazard_detected = True
        # planning_time_start = time.time()
        if hazard_detected or self.metrics['GoalReached'] or self.metrics['FakeGoalReached']:
            print('hazard_detected: {} or self.metrics[GoalReached]: {} or self.metrics[FakeGoalReached]: {}'
                  .format(hazard_detected, self.metrics['GoalReached'], self.metrics['FakeGoalReached']))
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            if self.planner_type == 'ImageBased':
                control = self._local_planner.run_step(grid_img=grid_img, debug=debug, grid_img_wo_ego=grid_img_wo_ego, img_step=img_step)
            else:
                control = self._local_planner.run_step(debug=debug)
        if self.num_view == 4 and self.args.baseline_type not in ['no_replan']:
            print('------------------- num_view=4 ------------------')
            print('img_step: {}, self.img_step_thres: {}'.format(img_step, self.img_step_thres))
            # pdb.set_trace()
            if self.img_step_thres[1] + 6 > img_step > self.img_step_thres[1] - 6:
                control.steer = -0.707
                # print('2 self.img_step_thres: ', self.img_step_thres)
                # pdb.set_trace()
                control.throttle = 0.7
            elif self.img_step_thres[2] > img_step >= self.img_step_thres[1] + 6:
                control.throttle = 0.45
                control.steer = -0.009
                print('1 control.steer: ', control)
            elif self.img_step_thres[3] - 8 > img_step >= self.img_step_thres[2] + 6:
                control.throttle = 0.45
                control.steer = -0.007
                print('2 control.steer: ', control)
            # elif self.img_step_thres[2] <= img_step:
            #     control.throttle = 0.7
            # elif 240 <= img_step: # < 270:
            #     control.brake = 1.
            # elif self.img_step_thres[3] + 20 > img_step > self.img_step_thres[3] - 6:
            elif self.img_step_thres[3] + 15 > img_step > self.img_step_thres[3] - 4:
                control.steer = -0.72
                # print('2 self.img_step_thres: ', self.img_step_thres)
                control.throttle = 0.7
                print('3 control.steer: ', control)
                # pdb.set_trace()
            elif self.img_step_thres[4] > img_step >= self.img_step_thres[3] + 13: #  + 20:
                control.throttle = 0.45
                control.steer = 0  # -0.007
                print('4 control.steer: ', control)
            elif img_step >= self.img_step_thres[4]: #  + 20:
                control.throttle = 0.7
                control.steer = 0.007
                print('4 control.steer: ', control)
            else: # if img_step <= self.img_step_thres[1] - 6
                control.steer = 0.
        planning_time_end = time.time()
        # planning_time = planning_time_end - planning_time_start
        # self.metrics['PlanningTime'].append(planning_time)
        if hasattr(self._local_planner, '_goal_reached'):
            self.metrics['GoalReached'] = self._local_planner._goal_reached
        if hasattr(self._local_planner, '_fake_goal_reached'):
            self.metrics['FakeGoalReached'] = self._local_planner._fake_goal_reached
        if not self.metrics['GoalReached']:
            self.metrics['TimestepsToReachGoals'] = self.step_index
            if hasattr(self._local_planner, '_receding_horizon'):
                self.metrics['TotalRecedingHorizon'] += self._local_planner._receding_horizon
        return control

    def done(self):
        """
        Check whether the agent has reached its destination.
        :return bool
        """
        return self._local_planner.done()

    def _is_vehicle_hazard_visual(self, vehicle_list, view_img, img_step=0, t2no_thres=70.,
                                  ego_vehicle_location_offset=[500, 866], pause_time=2, step_thres=10):
        """

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """
        if self.num_view == 2:
            if img_step < self.img_step_thres[0]:
                view_img_selected = view_img[0]
                ego_vehicle_location_offset = [75+128+500, 866 - 20]
            else:
                view_img_selected = view_img[1]
                ego_vehicle_location_offset = [75+128+500, 866 + 128 - 20]
        elif self.num_view == 4:
            # # camera height 13
            # if img_step < self.img_step_thres[0]:  # -1
            #     view_img_selected = view_img[0]
            #     ego_vehicle_location_offset = [75+128+500, 866 - 20]
            # elif img_step >= self.img_step_thres[0] and img_step < self.img_step_thres[1]:  # 0
            #     view_img_selected = view_img[1]
            #     ego_vehicle_location_offset = [75+128+500, 866 + 128 - 20]
            # elif img_step >= self.img_step_thres[1] and img_step < self.img_step_thres[2]:  # 1
            #     view_img_selected = view_img[1]
            #     ego_vehicle_location_offset = [75+128+500, 866 + 128 - 20]
            #     if self.exchange_flag == -1:
            #         self.forehead_distance = [self.forehead_distance[1], self.forehead_distance[0]]
            #         self.ego_car_size = [self.ego_car_size[1], self.ego_car_size[0]]
            #         print('self.forehead_distance: {}, self.ego_car_size: {}'.format(self.forehead_distance, self.ego_car_size))
            #         # pdb.set_trace()
            #         self.exchange_flag = 1
            # elif img_step >= self.img_step_thres[2] and img_step < self.img_step_thres[3]:  # 2
            #     view_img_selected = view_img[2]
            #     ego_vehicle_location_offset = [75+128+500+210, 866 + 128 - 20]
            # elif self.img_step_thres[4] > img_step >= self.img_step_thres[3]:  # 3
            #     view_img_selected = view_img[2]
            #     ego_vehicle_location_offset = [75+128+500+210, 866 + 128 - 20]
            #     if self.exchange_flag == 1:
            #         self.forehead_distance = [self.forehead_distance[1], self.forehead_distance[0]]
            #         self.ego_car_size = [self.ego_car_size[1], self.ego_car_size[0]]
            #         self.exchange_flag = 2
            #         print('self.forehead_distance: {}, self.ego_car_size: {}'.format(self.forehead_distance,
            #                                                                          self.ego_car_size))
            #         # pdb.set_trace()
            # elif img_step >= self.img_step_thres[4]:
            #     view_img_selected = view_img[3]
            #     ego_vehicle_location_offset = [75 + 128 + 500 + 210, 866 + 128 - 20 - 188]
            # camera height 15
            delta_view4_to_view2_view12 = 7
            delta_view4_to_view2_view34 = -2 + 15
            delta_view4_to_view2_view3_right = 20
            # [vertical (higher -> lower), horizontal (higher --> lefter)]
            if img_step < self.img_step_thres[0]:  # -1
                view_img_selected = view_img[0]
                ego_vehicle_location_offset = [75+128+500 - delta_view4_to_view2_view12, 866 - 20]
            elif img_step >= self.img_step_thres[0] and img_step < self.img_step_thres[1]:  # 0
                view_img_selected = view_img[1]
                ego_vehicle_location_offset = [75+128+500 - delta_view4_to_view2_view12, 866 + 128 - 20]
            elif img_step >= self.img_step_thres[1] and img_step < self.img_step_thres[2]:  # 1
                view_img_selected = view_img[1]
                ego_vehicle_location_offset = [75+128+500 - delta_view4_to_view2_view12 - 10 , 866 + 128 - 20]
                if self.exchange_flag == -1:
                    self.forehead_distance = [self.forehead_distance[1], self.forehead_distance[0]]
                    self.ego_car_size = [self.ego_car_size[1], self.ego_car_size[0]]
                    print('self.forehead_distance: {}, self.ego_car_size: {}'.format(self.forehead_distance, self.ego_car_size))
                    # pdb.set_trace()
                    self.exchange_flag = 1
            elif img_step >= self.img_step_thres[2] and img_step < self.img_step_thres[3]:  # 2
                view_img_selected = view_img[2]
                ego_vehicle_location_offset = [75+128+500+210 - delta_view4_to_view2_view34 + 5, 866 + 128 - 20 + 22 - 3 + 15]
            elif self.img_step_thres[4] > img_step >= self.img_step_thres[3]:  # 3
                view_img_selected = view_img[2]
                ego_vehicle_location_offset = [75+128+500+210 - delta_view4_to_view2_view34 + 13, 866 + 128 - 20 + delta_view4_to_view2_view3_right - 5 + 15]
                if self.exchange_flag == 1:
                    self.forehead_distance = [self.forehead_distance[1], self.forehead_distance[0]]
                    self.ego_car_size = [self.ego_car_size[1], self.ego_car_size[0]]
                    self.exchange_flag = 2
                    print('self.forehead_distance: {}, self.ego_car_size: {}'.format(self.forehead_distance,
                                                                                     self.ego_car_size))
                    # pdb.set_trace()
            elif img_step >= self.img_step_thres[4]:
                view_img_selected = view_img[3]
                ego_vehicle_location_offset = [75 + 128 + 500 + 210 - delta_view4_to_view2_view34 + 13, 866 + 128 - 20 - 120 + delta_view4_to_view2_view3_right - 70 - 15]

        ego_vehicle_location = self._vehicle.get_location()
        # print('1 ego_vehicle_location: ', ego_vehicle_location)
        # ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        view_img_selected = cv2.resize(view_img_selected, (int(128), int(128)))
        if self.save_img_dir is not None:
            cv2.imwrite(os.path.join(self.save_img_dir, 'rgb_{0:06d}.png'.format(self.step_index)), cv2.cvtColor(view_img_selected, cv2.COLOR_BGR2RGB))
            # print(len(view_img), view_img)
            # for i in view_img:
            #     print(i.shape)
            if self.num_view == 4:
                all_view_img = np.concatenate((cv2.resize(view_img[0], (128, 128)), np.ones_like(cv2.resize(view_img[0], (128, 128))[:, :4]),
                                               cv2.resize(view_img[1], (128, 128)), np.ones_like(cv2.resize(view_img[0], (128, 128))[:, :4]),
                                               cv2.resize(view_img[2], (128, 128)), np.ones_like(cv2.resize(view_img[0], (128, 128))[:, :4]),
                                               cv2.resize(view_img[3], (128, 128))), axis=1)
            elif self.num_view == 2:
                all_view_img = np.concatenate((cv2.resize(view_img[0], (128, 128)), np.ones_like(cv2.resize(view_img[0], (128, 128))[:, :4]), cv2.resize(view_img[1], (128, 128))), axis=1)
            else:
                raise NotImplementedError
            # all_view_img = cv2.cvtColor(all_view_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(self.save_img_dir, 'all_view_{0:06d}.png'.format(self.step_index)),
                        cv2.cvtColor(all_view_img, cv2.COLOR_BGR2RGB))
        frame = cv2.cvtColor(view_img_selected, cv2.COLOR_BGR2GRAY)  # BGR2GRAY
        # B = cv2.imread(
        #     os.path.join('../tools/carla_town02_8_view_20220524_color_waypoint', '_out_0', '00000240.png')).astype(
        #     np.uint8)
        # B = cv2.resize(B, (int(128), int(128)))
        # B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)  # BGR2GRAY
        # print('B.shape: {}, frame.shape: {}'.format(B.shape, frame.shape))  # (600, 600)
        if self.num_view == 2:
            if img_step < self.img_step_thres[0]:
                diff_img_T = (cv2.absdiff(self.B_0, frame) > t2no_thres) * 255.  # [False, True]
                diff_img_t2nd_T = (cv2.absdiff(self.B_0, frame) < t2no_thres) * 255.  # [False, True]
            else:
                diff_img_T = (cv2.absdiff(self.B_1, frame) > t2no_thres) * 255.  # [False, True]
                diff_img_t2nd_T = (cv2.absdiff(self.B_1, frame) < t2no_thres) * 255.  # [False, True]
        elif self.num_view == 4:
            if img_step < self.img_step_thres[0]:
                diff_img_T = (cv2.absdiff(self.B_0, frame) > t2no_thres) * 255.  # [False, True]
                diff_img_t2nd_T = (cv2.absdiff(self.B_0, frame) < t2no_thres) * 255.  # [False, True]
            elif img_step >= self.img_step_thres[0] and img_step < self.img_step_thres[1]:
                diff_img_T = (cv2.absdiff(self.B_1, frame) > t2no_thres) * 255.  # [False, True]
                diff_img_t2nd_T = (cv2.absdiff(self.B_1, frame) < t2no_thres) * 255.  # [False, True]
            elif img_step >= self.img_step_thres[1] and img_step < self.img_step_thres[2]:
                diff_img_T = (cv2.absdiff(self.B_2, frame) > t2no_thres) * 255.  # [False, True]
                diff_img_t2nd_T = (cv2.absdiff(self.B_2, frame) < t2no_thres) * 255.  # [False, True]
            elif img_step >= self.img_step_thres[2]:
                diff_img_T = (cv2.absdiff(self.B_2, frame) > t2no_thres) * 255.  # [False, True]
                diff_img_t2nd_T = (cv2.absdiff(self.B_2, frame) < t2no_thres) * 255.  # [False, True]

        # T2NO / T2ND
        if self.use_pytorch_model:
            with torch.no_grad():
                # if self.args.baseline_type == 'monolith':
                #     if self.planner_input_modality == 'frame':
                #         pass
                #     else:
                #         pass
                # else:
                SSTA_inference_time_start = time.time()
                # num_hidden = [int(x) for x in '16,16'.split(',')]
                # # x_batch = cv2.resize(view_img, (int(128), int(128)))
                x_batch = view_img_selected[None, None, :, :, :]
                # # print('x_batch.shape: ', x_batch.shape)  # (1, 1, 128, 128, 3)
                # batch = x_batch.shape[0]
                # height = x_batch.shape[2]
                # width = x_batch.shape[3]
                #
                # h_t_0 = []
                # c_t_0 = []
                # delta_c_list_0 = []
                # delta_m_list_0 = []
                #
                # for i in range(len(num_hidden)):
                #     zeros = torch.zeros([batch, num_hidden[i], height, width]).to(self.device)
                #     h_t_0.append(zeros)
                #     c_t_0.append(zeros)
                #     delta_c_list_0.append(zeros)
                #     delta_m_list_0.append(zeros)
                #
                # memory_0 = None  # torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
                # # x_0_t = x_batch
                # message_0 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(self.device)
                # message_1 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(self.device)
                # print('view_img.shape: ', view_img[0].shape, view_img[1].shape, ) # (600, 600, 3) (600, 600, 3)
                x_0_t_pred_prev = torch.tensor(cv2.resize(view_img[0][..., ::-1], (int(128), int(128)))[None, None, :, :, :]).float().to(self.device) / 255 # x_0_t[:, 0:1]
                x_1_t_pred_prev = torch.tensor(cv2.resize(view_img[1][..., ::-1], (int(128), int(128)))[None, None, :, :, :]).float().to(self.device) / 255  # x_0_t[:, 0:1]
                if self.num_view == 4:
                    x_2_t_pred_prev = torch.tensor(
                        cv2.resize(view_img[2][..., ::-1], (int(128), int(128)))[None, None, :, :, :]).float().to(
                        self.device) / 255  # x_0_t[:, 0:1]
                    x_3_t_pred_prev = torch.tensor(
                        cv2.resize(view_img[3][..., ::-1], (int(128), int(128)))[None, None, :, :, :]).float().to(
                        self.device) / 255  # x_0_t[:, 0:1]
                    cv2.imwrite('_out_0.png', view_img[0][..., ::-1])
                    cv2.imwrite('_out_1.png', view_img[1][..., ::-1])
                    cv2.imwrite('_out_2.png', view_img[2][..., ::-1])
                    cv2.imwrite('_out_3.png', view_img[3][..., ::-1])
                # x_0_t_pred, message_0, memory_0, h_t_0, c_t_0, delta_c_list_0, delta_m_list_0 = \
                #     self.model_0(x_0_t_pred_prev, message_0, message_1, memory_0, h_t_0, c_t_0, delta_c_list_0,
                #             delta_m_list_0)
                # print('x_0_t_pred_prev: ', x_0_t_pred_prev)
                baseline_type_in_no_replan_monolith = self.args.baseline_type in ['no_replan', 'monolith', 'nocomm', 'allcomm']
                if self.planner_input_modality == 'frame':
                    q_E = queue.Queue(self.his_length)
                    # print('x_0_t_pred_prev.shape: ', x_0_t_pred_prev.shape)
                    # print('2 x_0_t_pred_prev_input.shape: ', x_0_t_pred_prev_input.shape)
                    q_E.put(cv2.cvtColor(x_batch[0, 0], cv2.COLOR_BGR2GRAY))
                    x_0_t_pred_prev_input = x_0_t_pred_prev
                    x_1_t_pred_prev_input = x_1_t_pred_prev
                    if self.num_view == 4:
                        x_2_t_pred_prev_input = x_2_t_pred_prev
                        x_3_t_pred_prev_input = x_3_t_pred_prev
                    for t in range(self.his_length - 1):
                        # x_0_t_pred_prev_input = torch.cat(
                        #     (x_0_t_pred_prev, torch.zeros_like(x_0_t_pred_prev)[:, :, :, :, :2]),
                        #     axis=-1)  # (1, 1, 128, 128, 3)
                        message_0 = self.vae.get_message(x_0_t_pred_prev_input).detach()
                        message_1 = self.vae.get_message(x_1_t_pred_prev_input).detach()
                        if self.args.baseline_type in ['nocomm']:
                            # message_0 = torch.zeros_like(message_0)
                            message_1 = torch.zeros_like(message_1)
                        if self.args.delay_time == 'sim' and not baseline_type_in_no_replan_monolith:
                            # SSTA Broadcast Rate: 4Hz, 0.25s
                            time.sleep(0.25)
                        if not baseline_type_in_no_replan_monolith:
                            if self.num_view == 2:
                                x_0_t_pred_prev_input, _, self.memory_0 = self.model_0(x_0_t_pred_prev_input, message_0, message_1, self.memory_0)
                            elif self.num_view == 4:
                                x_0_t_pred_prev_input, _, self.memory_0 = self.model_0(x_0_t_pred_prev_input, message_0, message_1, self.memory_0)
                            x_1_t_pred_prev_input, _, self.memory_1 = self.model_1(x_1_t_pred_prev_input, message_1,
                                                                                   message_0, self.memory_1)
                        if self.args.delay_time == 'sim' and not baseline_type_in_no_replan_monolith:
                            # SSTA Broadcast Rate: 4Hz, 0.25s
                            time.sleep(0.25)
                        # if not baseline_type_in_no_replan_monolith:
                        #     x_1_t_pred_prev_input, _, self.memory_1 = self.model_1(x_1_t_pred_prev_input, message_1, message_0, self.memory_1)
                        if baseline_type_in_no_replan_monolith:
                            if self.num_view == 2:
                                x_t_pred_prev_input = torch.cat((x_0_t_pred_prev_input, x_1_t_pred_prev_input), axis=-1)
                                x_t_pred_prev_input, _, self.memory_0 = self.model_0(x_t_pred_prev_input, None,
                                                                                       None, self.memory_0)
                                x_0_t_pred_prev_input, x_1_t_pred_prev_input = torch.split(x_t_pred_prev_input, x_t_pred_prev_input.shape[-1] // 2, dim=-1)
                            elif self.num_view == 4:
                                x_t_pred_prev_input = torch.cat((x_0_t_pred_prev_input, x_1_t_pred_prev_input, x_2_t_pred_prev_input, x_3_t_pred_prev_input), axis=-1)
                                x_t_pred_prev_input, _, self.memory_0 = self.model_0(x_t_pred_prev_input, None,
                                                                                     None, self.memory_0)
                                x_0_t_pred_prev_input, x_1_t_pred_prev_input, x_2_t_pred_prev_input, x_3_t_pred_prev_input = torch.split(x_t_pred_prev_input,
                                                                                           x_t_pred_prev_input.shape[
                                                                                               -1] // self.num_view, dim=-1)
                        # print('x_0_t_pred_prev_input[0, 0].detach().cpu().numpy().shape: ', x_0_t_pred_prev_input[0, 0].detach().cpu().numpy().shape,
                        #       B.shape, self.B.max(), cv2.cvtColor(x_0_t_pred_prev_input[0, 0].detach().cpu().numpy(), cv2.COLOR_BGR2GRAY).shape,
                        #       cv2.cvtColor(x_0_t_pred_prev_input[0, 0].detach().cpu().numpy() * 255., cv2.COLOR_BGR2GRAY).max())
                        x_0_t_pred_prev_input_int = cv2.cvtColor((x_0_t_pred_prev_input[0, 0].detach().cpu().numpy() * 255.).astype(np.uint8), cv2.COLOR_BGR2RGB)
                        x_1_t_pred_prev_input_int = cv2.cvtColor((x_1_t_pred_prev_input[0, 0].detach().cpu().numpy() * 255.).astype(np.uint8), cv2.COLOR_BGR2RGB)
                        gray_x_0_t_pred_prev_input = cv2.cvtColor(x_0_t_pred_prev_input_int, cv2.COLOR_BGR2GRAY)
                        gray_x_1_t_pred_prev_input = cv2.cvtColor(x_1_t_pred_prev_input_int, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(os.path.join(self.save_img_dir, 'pred_frame_{:02d}_{:06d}_{:02d}.png'.format(0, self.step_index, t)), x_0_t_pred_prev_input_int)
                        cv2.imwrite(os.path.join(self.save_img_dir, 'pred_frame_{:02d}_{:06d}_{:02d}.png'.format(1, self.step_index, t)), x_1_t_pred_prev_input_int)
                        if self.num_view == 4:
                            x_2_t_pred_prev_input_int = cv2.cvtColor(
                                (x_2_t_pred_prev_input[0, 0].detach().cpu().numpy() * 255.).astype(np.uint8),
                                cv2.COLOR_BGR2RGB)
                            x_3_t_pred_prev_input_int = cv2.cvtColor(
                                (x_3_t_pred_prev_input[0, 0].detach().cpu().numpy() * 255.).astype(np.uint8),
                                cv2.COLOR_BGR2RGB)
                            gray_x_2_t_pred_prev_input = cv2.cvtColor(x_2_t_pred_prev_input_int, cv2.COLOR_BGR2GRAY)
                            gray_x_3_t_pred_prev_input = cv2.cvtColor(x_3_t_pred_prev_input_int, cv2.COLOR_BGR2GRAY)
                            cv2.imwrite(os.path.join(self.save_img_dir,
                                                     'pred_frame_{:02d}_{:06d}_{:02d}.png'.format(2, self.step_index,
                                                                                                  t)),
                                                     x_2_t_pred_prev_input_int)
                            cv2.imwrite(os.path.join(self.save_img_dir,
                                                     'pred_frame_{:02d}_{:06d}_{:02d}.png'.format(3, self.step_index,
                                                                                                  t)),
                                                     x_3_t_pred_prev_input_int)
                        if self.num_view == 2:
                            if img_step < self.img_step_thres[0]:
                                # print('self.B.shape, gray_x_0_t_pred_prev_input.shape: ', self.B.shape, gray_x_0_t_pred_prev_input.shape)
                                diff_img_T = (cv2.absdiff(self.B_0, gray_x_0_t_pred_prev_input) > t2no_thres) * 255.  # [False, True]
                            else:
                                if img_step == self.img_step_thres[0]:
                                    q_E = queue.Queue(self.his_length)
                                # print('self.B.shape, gray_x_0_t_pred_prev_input.shape: ', self.B.shape, gray_x_0_t_pred_prev_input.shape)
                                diff_img_T = (cv2.absdiff(self.B_1, gray_x_1_t_pred_prev_input) > t2no_thres) * 255.  # [False, True]
                            # diff_img_t2nd_T = (cv2.absdiff(B, cv2.cvtColor(x_0_t_pred_prev_input[0, 0], cv2.COLOR_BGR2GRAY)) < thres) * 255.  # [False, True]
                        elif self.num_view == 4:
                            if img_step < self.img_step_thres[0]:
                                # print('self.B.shape, gray_x_0_t_pred_prev_input.shape: ', self.B.shape, gray_x_0_t_pred_prev_input.shape)
                                diff_img_T = (cv2.absdiff(self.B_0, gray_x_0_t_pred_prev_input) > t2no_thres) * 255.  # [False, True]
                            elif img_step >= self.img_step_thres[0] and img_step < self.img_step_thres[2]:
                                if img_step == self.img_step_thres[0]:
                                    q_E = queue.Queue(self.his_length)
                                # print('self.B.shape, gray_x_0_t_pred_prev_input.shape: ', self.B.shape, gray_x_0_t_pred_prev_input.shape)
                                diff_img_T = (cv2.absdiff(self.B_1, gray_x_1_t_pred_prev_input) > t2no_thres) * 255.  # [False, True]
                            elif img_step >= self.img_step_thres[2] and img_step < self.img_step_thres[4]:
                                if img_step == self.img_step_thres[1]:
                                    q_E = queue.Queue(self.his_length)
                                # print('self.B.shape, gray_x_0_t_pred_prev_input.shape: ', self.B.shape, gray_x_0_t_pred_prev_input.shape)
                                diff_img_T = (cv2.absdiff(self.B_2, gray_x_1_t_pred_prev_input) > t2no_thres) * 255.  # [False, True]
                            elif img_step >= self.img_step_thres[4]: # and img_step < self.img_step_thres[5]:
                                if img_step == self.img_step_thres[3]:
                                    q_E = queue.Queue(self.his_length)
                                # print('self.B.shape, gray_x_0_t_pred_prev_input.shape: ', self.B.shape, gray_x_0_t_pred_prev_input.shape)
                                diff_img_T = (cv2.absdiff(self.B_3, gray_x_1_t_pred_prev_input) > t2no_thres) * 255.  # [False, True]
                        while not q_E.full():
                            q_E.put(diff_img_T)

                    x_0_t_pred = np.asarray(
                        [ele for ele in list(q_E.queue)] + [np.ones_like(diff_img_T) * 255.])[None, ]
                    self.memory_0 = None
                    # print('x_0_t_pred.shape: ', x_0_t_pred.shape)  #  (1, 11, 128, 128)
                    t2no_img = np.argmax(x_0_t_pred[0, ], axis=0)  # [:, :, None]
                else:  # T2NO/T2ND
                    message_0 = self.vae.get_message(x_0_t_pred_prev).detach()
                    message_1 = self.vae.get_message(x_1_t_pred_prev).detach()
                    if self.args.baseline_type in ['nocomm']:
                        # message_0 = torch.zeros_like(message_0)
                        message_1 = torch.zeros_like(message_1)
                        print('x_0_t_pred_prev_input.shape: ', x_0_t_pred_prev.shape)  # [1, 1, 128, 128, 3]
                        # x_0_t_pred_prev[..., -1] = torch.zeros_like(x_0_t_pred_prev[..., -1])
                        # x_0_t_pred_prev[:, :, :64, :64, :] = torch.zeros_like(x_0_t_pred_prev[:, :, :64, :64, :])
                        x_0_t_pred_prev[:, :, :100, :100, :] = torch.zeros_like(x_0_t_pred_prev[:, :, :100, :100, :])
                    if self.args.delay_time == 'sim' and baseline_type_in_no_replan_monolith:
                        # SSTA Broadcast Rate: 4Hz, 0.25s
                        time.sleep(0.25)
                    if not baseline_type_in_no_replan_monolith:
                        x_0_t_pred = self.model_0(x_0_t_pred_prev, message_0, message_1)
                        x_1_t_pred = self.model_1(x_1_t_pred_prev, message_1, message_0)
                    if self.args.delay_time == 'sim' and baseline_type_in_no_replan_monolith:
                        # SSTA Broadcast Rate: 4Hz, 0.25s
                        time.sleep(0.25)
                    # if not baseline_type_in_no_replan_monolith:
                    #     x_1_t_pred = self.model_1(x_1_t_pred_prev, message_1, message_0)
                    if baseline_type_in_no_replan_monolith:
                        if self.num_view == 4:
                            x_t_pred_prev = torch.cat((x_0_t_pred_prev, x_1_t_pred_prev, x_2_t_pred_prev, x_3_t_pred_prev), axis=-1)

                            if self.args.baseline_type in ['allcomm']:
                                message_0 = self.vae.get_message(x_0_t_pred_prev).detach()
                                message_1 = self.vae.get_message(x_1_t_pred_prev).detach()
                                message_2 = self.vae.get_message(x_2_t_pred_prev).detach()
                                message_3 = self.vae.get_message(x_3_t_pred_prev).detach()
                                print('x_t_pred_prev.shape, message_3.shape: ', x_t_pred_prev.shape, message_3.shape)
                                # x_t_pred_prev.shape, message_3.shape:  torch.Size([1, 1, 128, 128, 12]) torch.Size([1, 1, 128, 128, 1])
                                '''
                                x_t_pred_prev = torch.cat((x_t_pred_prev, message_0, message_1, message_2, message_3), dim=-1)
                            '''
                            # x_t_pred = self.model_0(x_t_pred_prev, None, None, None, None)
                            # x_0_t_pred, x_1_t_pred, x_2_t_pred, x_3_t_pred = torch.split(x_t_pred,
                            #                                                              x_t_pred.shape[1] // 4, dim=1)
                            try:
                                x_t_pred, _, _ = self.model_0(x_t_pred_prev, None, None, None)
                                # print('1 x_t_pred.shape: ', x_t_pred.shape)
                                x_t_pred = x_t_pred[:, 0].permute(0, 3, 1, 2)
                                x_0_t_pred, x_1_t_pred, x_2_t_pred, x_3_t_pred = torch.split(x_t_pred,
                                                                                             x_t_pred.shape[1] // 4,
                                                                                             dim=1)
                            except:
                                x_t_pred = self.model_0(x_t_pred_prev, None, None, None, None)
                                # print('1 x_t_pred.shape: ', x_t_pred.shape)
                                # x_t_pred = x_t_pred[:, 0].permute(0, 3, 1, 2)
                                x_0_t_pred, x_1_t_pred, x_2_t_pred, x_3_t_pred = torch.split(x_t_pred,
                                                                                             x_t_pred.shape[1] // 4,
                                                                                             dim=1)
                            # print('2 x_t_pred.shape: ', x_t_pred.shape)
                            # pdb.set_trace()
                        elif self.num_view == 2:
                            x_t_pred_prev = torch.cat((x_0_t_pred_prev, x_1_t_pred_prev), axis=-1)
                            x_t_pred = self.model_0(x_t_pred_prev, None, None)
                            x_0_t_pred, x_1_t_pred = torch.split(x_t_pred, x_t_pred.shape[1] // 2, dim=1)
                # x_0_t_pred_prev = x_0_t_pred.detach()
                # print('x_0_t_pred.shape: ', x_0_t_pred.shape)
                # x_0_t_pred.shape:  torch.Size([1, 11, 128, 128])
                    if self.num_view == 2:
                        if img_step < self.img_step_thres[0]:
                            t2no_img = np.argmax(x_0_t_pred[0,].detach().cpu().numpy(), axis=0)  # [:, :, None]
                        else:
                            t2no_img = np.argmax(x_1_t_pred[0,].detach().cpu().numpy(), axis=0) # [:, :, None]
                    elif self.num_view == 4:
                        if img_step < self.img_step_thres[0]:
                            t2no_img = np.argmax(x_0_t_pred[0,].detach().cpu().numpy(), axis=0)  # [:, :, None]
                        elif img_step < self.img_step_thres[2]:
                            t2no_img = np.argmax(x_1_t_pred[0,].detach().cpu().numpy(), axis=0)  # [:, :, None]
                        elif img_step < self.img_step_thres[4]:
                            t2no_img = np.argmax(x_2_t_pred[0,].detach().cpu().numpy(), axis=0)  # [:, :, None]
                        else:  # if img_step < self.img_step_thres[5]:
                            t2no_img = np.argmax(x_3_t_pred[0,].detach().cpu().numpy(), axis=0)  # [:, :, None]
                        # else:
                        #     t2no_img = np.argmax(x_1_t_pred[0,].detach().cpu().numpy(), axis=0) # [:, :, None]
                    # for i in range(4):
                    #     plt.imshow(view_img[i])
                    #     plt.show()
                    
                    # print('t2no_img.max(), t2no_img.min(): ', t2no_img.max(), t2no_img.min())
                    # plt.imshow(x_1_t_pred_prev[0, 0].cpu())
                    # plt.show()
                    # plt.imshow(t2no_img)
                    # plt.show()
                # print('t2no_img.shape: ', t2no_img.shape)
                # print('t2no_img.shape: ', t2no_img.shape, t2no_img.max(), t2no_img.min())  # (128, 128) 1 1
                # t2no_img = cv2.resize(t2no_img.astype('float32'), (int(600), int(600)))
                SSTA_inference_time_end = time.time()
                SSTA_inference_time = SSTA_inference_time_end - SSTA_inference_time_start
                print('****** SSTA_inference_time: {}'.format(SSTA_inference_time))
                if self.vis and self.step_index > self.start_vis_frame_index:
                    # t2nd_img_vis = ((10 - t2no_img) * 255. / (self.his_length + 1)).astype('uint8')
                    t2no_img_vis = (t2no_img * 255. / (self.his_length + 1)).astype('uint8')
                    t2nd_img_vis = np.zeros_like(t2no_img_vis).astype('uint8')
                    # t2no_img_vis = ((10 - t2no_img) * 255 / self.his_length).astype('uint8')
                    # t2nd_img_vis = (t2no_img * 255 / self.his_length).astype('uint8')

                    # print('B.shape, frame.shape, diff_img.shape, diff_img_t2nd.shape, t2no_img.shape, t2nd_img.shape: ',
                    #       B.shape, frame.shape, diff_img.shape, diff_img_t2nd.shape, t2no_img.shape, t2nd_img.shape)
                    concat_img = np.concatenate(
                        [B, np.zeros_like(B[:, :4]), frame, np.zeros_like(B[:, :4]), diff_img_T, np.zeros_like(B[:, :4]),
                         diff_img_t2nd_T, np.zeros_like(B[:, :4]), t2no_img_vis, np.zeros_like(B[:, :4]), t2nd_img_vis], axis=1)

                    final = np.concatenate((self.img_title, concat_img), axis=0)
                    # print('final.shape: ', concat_img.shape)  # final.shape:  (600, 3620)
                    final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                    plt.figure(figsize=(30 * 4, 30))
                    plt.title('BG, Frame T, Diff_T2NO, Diff_T2ND, T2NO, T2ND')
                    plt.imshow(final, cmap='gray', )
                    plt.show()
                    if pause_time is not None:
                        plt.ion()
                        plt.pause(pause_time)
                        plt.close('all')
                if self.save_img_dir is not None:
                    # t2nd_img_vis = ((10 - t2no_img) * 255. / (self.his_length + 1)).astype('uint8')
                    t2no_img_vis = (t2no_img * 255. / (self.his_length + 1)).astype('uint8')
                    t2nd_img_vis = np.zeros_like(t2no_img_vis).astype('uint8')
                    print('img_step: ', img_step)
                    if img_step < self.img_step_thres[0]:
                        concat_img = np.concatenate(
                            [self.B_0, np.zeros_like(self.B_0[:, :4]), frame, np.zeros_like(self.B_0[:, :4]), diff_img_T, np.zeros_like(self.B_0[:, :4]),
                             diff_img_t2nd_T, np.zeros_like(self.B_0[:, :4]), t2no_img_vis, np.zeros_like(self.B_0[:, :4]), t2nd_img_vis],
                            axis=1)
                    else:
                        concat_img = np.concatenate(
                            [self.B_1, np.zeros_like(self.B_1[:, :4]), frame, np.zeros_like(self.B_1[:, :4]), diff_img_T,
                             np.zeros_like(self.B_1[:, :4]),
                             diff_img_t2nd_T, np.zeros_like(self.B_1[:, :4]), t2no_img_vis, np.zeros_like(self.B_1[:, :4]), t2nd_img_vis],
                            axis=1)

                    final = np.concatenate((self.img_title, concat_img), axis=0)
                    final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                    cv2.imwrite(os.path.join(self.save_img_dir, 't2no_{0:06d}.png'.format(self.step_index)), final)
        else:
            if self.t2no_t2nd_method == 'v1':
                # while not self.q_E.full():
                #     self.q_E.put(np.zeros_like(diff_img))
                while not self.q_E_t2nd.full():
                    self.q_E_t2nd.put(diff_img_t2nd_T)
                    # self.q_E_t2nd.put(np.zeros_like(diff_img_t2nd))

                q_E_t2nd_array = np.asarray([ele for ele in reversed(list(self.q_E_t2nd.queue))] + [np.ones_like(diff_img_t2nd_T)])
                t2nd_img = np.argmax(np.asarray([ele for ele in reversed(list(self.q_E_t2nd.queue))] + [np.ones_like(diff_img_t2nd_T)]), axis=0)
                print('self.step_index: {}, (t2nd_img.min(): {}, t2nd_img.max(): {})'.format(self.step_index, t2nd_img.min(), t2nd_img.max()))  # 1, (0, self.his_length)
                t2no_img = q_E_t2nd_array.shape[0] - t2nd_img
                print('self.step_index: {}, (t2no_img.min(): {}, t2no_img.max(): {})'.format(self.step_index, t2no_img.min(), t2no_img.max()))  # 1, 2

                if self.vis and self.step_index > self.start_vis_frame_index:
                    t2nd_img_vis = (t2nd_img * 255 / self.his_length).astype('uint8')
                    t2no_img_vis = (t2no_img * 255 / self.his_length).astype('uint8')

                # if self.step_index == 0:
                #     self.q_E.get()
                #     # self.q_E_t2nd.get()
                #     self.q_E.put(diff_img)
                #     # self.q_E_t2nd.put(diff_img_t2nd)
                #     # t2no_img = (np.argmax(np.asarray([ele for ele in list(self.q_E.queue)]), axis=0)) * int(
                #     #     255 / self.his_length)
                #     q_E_array = np.asarray([ele for ele in reversed(list(self.q_E.queue))])
                #     t2no_img = np.argmax(q_E_array, axis=0)
                #     t2no_img = (np.ones_like(t2no_img) * q_E_array.shape[0] - t2no_img)
                #     print('self.step_index, t2no_img.max(): ', self.step_index, t2no_img.max())  # (0, 10)
                #     # self.first_t2no_img = t2no_img
                #     if self.vis:
                #         t2no_img = (t2no_img * 255 / self.his_length).astype('uint8')
                # else:
                #     # t2no_img_binary = np.argmax(np.asarray([ele for ele in list(self.q_E.queue)]), axis=0)
                #     q_E_array = np.asarray([ele for ele in reversed(list(self.q_E.queue))])
                #     t2no_img_argmax = np.argmax(q_E_array, axis=0)
                #     t2no_img_binary = (np.ones_like(t2no_img_argmax) * q_E_array.shape[0] - t2no_img_argmax)  #  * int(255 / self.his_length)
                #     t2no_img = (t2no_img_binary > 0) * (
                #                 t2no_img_binary - self.his_length + min(self.step_index, self.his_length - 1) + 1)
                #     mask = (t2no_img == (self.step_index + 1))
                #     print('mask: ', mask)
                #     t2no_img[mask] = self.his_length
                #     # t2no_img = np.minimum(t2no_img, self.first_t2no_img)
                #     print('self.step_index, t2no_img.max(): ', self.step_index, t2no_img.max()) # 1, 2
                #     if self.vis:
                #         t2no_img = (t2no_img * 255 / self.his_length).astype('uint8')

                # print('t2no_img.shape: ', t2no_img.shape, 'diff_img.shape: ', diff_img.shape,
                #       'diff_img.max(): ', diff_img.max(), 'diff_img.min(): ', diff_img.min())
                # t2no_img.shape:  (600, 600) diff_img.shape:  (600, 600) diff_img.max():  255.0 diff_img.min():  0.0
                # bg_error[each_img] = np.sum(diff_img).tolist()
                # cv2.imwrite(os.path.join(save_dir, each_view, 'diff_' + each_img, ), diff_img)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'diffT2ND_' + each_img, ), diff_img_t2nd)
                # cv2.imwrite(os.path.join(save_dir, each_view, 't2no_' + each_img, ), t2no_img)
                # cv2.imwrite(os.path.join(save_dir, each_view, 't2nd_' + each_img, ), t2nd_img)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'frame_' + each_img, ), frame)
                # if mode not in ['static', 'mean']:
                #     cv2.imwrite(os.path.join(save_dir, each_view, 'mask_' + each_img, ), inverse_mask)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'bg_' + each_img, ), B)
                # print('Save to {}, diff error: {}'.format(os.path.join(save_dir, each_view, each_img, ), bg_error))
                    concat_img = np.concatenate([B, np.zeros_like(B[:, :4]), frame, np.zeros_like(B[:, :4]), diff_img_T, np.zeros_like(B[:, :4]),
                                                 diff_img_t2nd_T, np.zeros_like(B[:, :4]), t2no_img_vis, np.zeros_like(B[:, :4]), t2nd_img_vis], axis=1)
                    print('self.img_title.shape, concat_img.shape: ', self.img_title.shape, concat_img.shape)
                    final = np.concatenate((self.img_title, concat_img), axis=0)
                    final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                    # cv2.imshow('Image', final)
                    # cv2.waitKey()
                    plt.figure(figsize=(30 * 4, 30))
                    plt.title('BG, Frame T, Diff_T2NO, Diff_T2ND, T2NO, T2ND')
                    plt.imshow(final, cmap='gray',)
                    plt.show()
                    if pause_time is not None:
                        plt.ion()
                        plt.pause(pause_time)
                        plt.close('all')
                self.q_E_t2nd.get()
            else:  # self.t2no_t2nd_method == 'v2':
                while not self.q_E.full():
                    self.q_E.put(diff_img_T)
                while not self.q_E_t2nd.full():
                    self.q_E_t2nd.put(diff_img_t2nd_T)
                    # self.q_E_t2nd.put(np.zeros_like(diff_img_t2nd))
                # self.q_E.get()
                # self.q_E.put(diff_img)

                q_E_t2no_array = np.asarray(
                    [ele for ele in list(self.q_E.queue)] + [np.ones_like(diff_img_T) * 255.])
                t2no_img = np.argmax(q_E_t2no_array, axis=0)
                print('self.step_index: {}, (t2no_img.min(): {}, t2no_img.max(): {})'.format(self.step_index,
                                                                                             t2no_img.min(),
                                                                                             t2no_img.max()))  # 1, 2

                q_E_t2nd_array = np.asarray([ele for ele in list(self.q_E_t2nd.queue)])  # + [np.ones_like(diff_img_t2nd)])
                t2nd_img = np.argmax(q_E_t2nd_array, axis=0)
                print('self.step_index: {}, (t2nd_img.min(): {}, t2nd_img.max(): {})'.format(self.step_index,
                                                                                             t2nd_img.min(),
                                                                                             t2nd_img.max()))  # 1, (0, self.his_length)
                infty_mask = np.logical_or((np.abs(t2no_img - self.his_length) < 1e-2), (np.abs(t2nd_img) < 1e-2))
                # print('np.sum(infty_mask): ', np.sum(infty_mask))
                t2nd_img[infty_mask] = self.his_length
                # t2no_img = q_E_t2nd_array.shape[0] - t2nd_img

                self.q_E.get()
                diff_img_t2nd_t = self.q_E_t2nd.get()

                if self.vis and self.step_index > self.start_vis_frame_index:
                    t2nd_img_vis = (t2nd_img * 255 / (self.his_length + 1)).astype('uint8')
                    t2no_img_vis = (t2no_img * 255 / (self.his_length + 1)).astype('uint8')

                    # if self.step_index == 0:
                    #     self.q_E.get()
                    #     # self.q_E_t2nd.get()
                    #     self.q_E.put(diff_img)
                    #     # self.q_E_t2nd.put(diff_img_t2nd)
                    #     # t2no_img = (np.argmax(np.asarray([ele for ele in list(self.q_E.queue)]), axis=0)) * int(
                    #     #     255 / self.his_length)
                    #     q_E_array = np.asarray([ele for ele in reversed(list(self.q_E.queue))])
                    #     t2no_img = np.argmax(q_E_array, axis=0)
                    #     t2no_img = (np.ones_like(t2no_img) * q_E_array.shape[0] - t2no_img)
                    #     print('self.step_index, t2no_img.max(): ', self.step_index, t2no_img.max())  # (0, 10)
                    #     # self.first_t2no_img = t2no_img
                    #     if self.vis:
                    #         t2no_img = (t2no_img * 255 / self.his_length).astype('uint8')
                    # else:
                    #     # t2no_img_binary = np.argmax(np.asarray([ele for ele in list(self.q_E.queue)]), axis=0)
                    #     q_E_array = np.asarray([ele for ele in reversed(list(self.q_E.queue))])
                    #     t2no_img_argmax = np.argmax(q_E_array, axis=0)
                    #     t2no_img_binary = (np.ones_like(t2no_img_argmax) * q_E_array.shape[0] - t2no_img_argmax)  #  * int(255 / self.his_length)
                    #     t2no_img = (t2no_img_binary > 0) * (
                    #                 t2no_img_binary - self.his_length + min(self.step_index, self.his_length - 1) + 1)
                    #     mask = (t2no_img == (self.step_index + 1))
                    #     print('mask: ', mask)
                    #     t2no_img[mask] = self.his_length
                    #     # t2no_img = np.minimum(t2no_img, self.first_t2no_img)
                    #     print('self.step_index, t2no_img.max(): ', self.step_index, t2no_img.max()) # 1, 2
                    #     if self.vis:
                    #         t2no_img = (t2no_img * 255 / self.his_length).astype('uint8')

                    # print('t2no_img.shape: ', t2no_img.shape, 'diff_img.shape: ', diff_img.shape,
                    #       'diff_img.max(): ', diff_img.max(), 'diff_img.min(): ', diff_img.min())
                    # t2no_img.shape:  (600, 600) diff_img.shape:  (600, 600) diff_img.max():  255.0 diff_img.min():  0.0
                    # bg_error[each_img] = np.sum(diff_img).tolist()
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'diff_' + each_img, ), diff_img)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'diffT2ND_' + each_img, ), diff_img_t2nd)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 't2no_' + each_img, ), t2no_img)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 't2nd_' + each_img, ), t2nd_img)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'frame_' + each_img, ), frame)
                    # if mode not in ['static', 'mean']:
                    #     cv2.imwrite(os.path.join(save_dir, each_view, 'mask_' + each_img, ), inverse_mask)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'bg_' + each_img, ), B)
                    # print('Save to {}, diff error: {}'.format(os.path.join(save_dir, each_view, each_img, ), bg_error))
                    concat_img = np.concatenate(
                        [B, np.zeros_like(B[:, :4]), frame, np.zeros_like(B[:, :4]), diff_img_T,
                         np.zeros_like(B[:, :4]),
                         diff_img_t2nd_T, np.zeros_like(B[:, :4]), t2no_img_vis, np.zeros_like(B[:, :4]),
                         t2nd_img_vis], axis=1)
                    print('self.img_title.shape, concat_img.shape: ', self.img_title.shape, concat_img.shape)
                    final = np.concatenate((self.img_title, concat_img), axis=0)
                    final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                    # cv2.imshow('Image', final)
                    # cv2.waitKey()
                    plt.figure(figsize=(30 * 4, 30))
                    plt.title('BG, Frame T, Diff_T2NO, Diff_T2ND, T2NO, T2ND')
                    plt.imshow(final, cmap='gray', )
                    plt.show()
                    if pause_time is not None:
                        plt.ion()
                        plt.pause(pause_time)
                        plt.close('all')

            if self.save_img_dir is not None:
                # t2nd_img_vis = ((10 - t2no_img) * 255 / self.his_length).astype('uint8')
                t2nd_img_vis = (t2nd_img * 255 / (self.his_length + 1)).astype('uint8')
                t2no_img_vis = (t2no_img * 255 / (self.his_length + 1)).astype('uint8')
                concat_img = np.concatenate(
                    [B, np.zeros_like(B[:, :4]), frame, np.zeros_like(B[:, :4]), diff_img_T, np.zeros_like(B[:, :4]),
                     diff_img_t2nd_T, np.zeros_like(B[:, :4]), t2no_img_vis, np.zeros_like(B[:, :4]), t2nd_img_vis],
                    axis=1)

                final = np.concatenate((self.img_title, concat_img), axis=0)
                final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                cv2.imwrite(os.path.join(self.save_img_dir, 't2no_{0:06d}.png'.format(self.step_index)), final)


        # for target_vehicle in vehicle_list:
        #     # do not account for the ego vehicle
        #     if target_vehicle.id == self._vehicle.id:
        #         continue
        #
        #     # if the object is not in our lane it's not an obstacle
        #     target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
        #     if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
        #             target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
        #         continue
        #
        #     if is_within_distance_ahead(target_vehicle.get_transform(),
        #                                 self._vehicle.get_transform(),
        #                                 self._proximity_vehicle_threshold):
        #         return (True, target_vehicle)

        # if is_within_distance_ahead(target_vehicle.get_transform(), self._vehicle.get_transform(),
        #                                 self._proximity_vehicle_threshold):
        #     return (True, 0)
        # else:
        if self.collision_detection_mode == 'contour':  # 'contour'
            # center of the contour detection
            # https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
            # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
            diff_img_3_channel = np.float32(np.concatenate((t2no_img[:, :, None], t2no_img[:, :, None], t2no_img[:, :, None]), axis=2))
            # print('diff_img_3_channel.shape: ', diff_img_3_channel.shape, diff_img_3_channel.max(), diff_img_3_channel.min())
            # gray = cv2.cvtColor(diff_img_3_channel, cv2.COLOR_BGR2GRAY)
            gray = t2no_img
            kernel = np.ones((5, 5), dtype=np.uint8)
            gray = cv2.dilate(gray, kernel, 6)
            gray = cv2.erode(gray, kernel, iterations=4)
            blur = cv2.GaussianBlur(gray, (63, 63), cv2.BORDER_DEFAULT)
            ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
            # bg_thresh = np.zeros_like(thresh)
            # bg_thresh[10:-10, 10:-10] = thresh[10:-10, 10:-10]
            # thresh = bg_thresh
            # print('blur.max(), blur.min(): ', blur.max(), blur.min())
            # plt.imshow(blur)
            # plt.show()
            # print('thresh.max(), thresh.min(): ', thresh.max(), thresh.min())
            # plt.imshow(thresh)
            # plt.show()
            contours, hierarchies = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # retval, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh.astype(np.uint8), connectivity=4)
            blank = np.zeros(thresh.shape[:2], dtype='uint8')
            cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)
            center_list = []
            for idx, i in enumerate(contours):
                M = cv2.moments(i)
                area = cv2.contourArea(i)
                print('area: ', area)
                if M['m00'] != 0:
                    # print('M: ', M)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    skip_flag = False
                    if np.sum(np.abs(cx - 300) + np.abs(cy - 300)) < 20:
                        print(f"skip x: {cx} y: {cy}")
                        skip_flag = True
                    for (prev_x, prev_y) in center_list:
                        if np.sum(np.abs(cx - prev_x) + np.abs(cy - prev_y)) < 80:
                            print(f"skip x: {cx} y: {cy}")
                            skip_flag = True
                    if not skip_flag:
                        center_list.append((cx, cy))
                        cv2.drawContours(diff_img_3_channel, [i], -1, (0, 255, 0), 2)
                        cv2.circle(diff_img_3_channel, (cx, cy), 7, (0, 0, 255), -1)
                        cv2.putText(diff_img_3_channel, "center", (cx - 20, cy - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        print(f"x: {cx} y: {cy}")
            # for (i, label) in enumerate(labels):
            #     if label == 0:  # background
            #         continue
            #     numPixels = stats[i][-1]
            #     cx, cy = int(centroids[i][0]), int(centroids[i][1])
            #     print('centroids[i]: ', centroids[i])
            #     if numPixels > 0:
            #         cv2.circle(diff_img_3_channel, (cx, cy), 7, (0, 0, 255), -1)
            #         cv2.putText(diff_img_3_channel, "center", (cx - 20, cy - 20),
            #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #         center_list.append((cx, cy))
            # concat_img = np.concatenate(
            #     [blur, np.zeros_like(blur[:, :4]), thresh, np.zeros_like(thresh[:, :4]), diff_img_3_channel[:, :, 0]], axis=1)
            if self.vis and self.step_index > self.start_vis_frame_index:
                blur = np.repeat(blur[:, :, None], 3, axis=-1)
                thresh = np.repeat(thresh[:, :, None], 3, axis=-1)
                print('blur.shape, thresh.shape, diff_img_3_channel.shape: ', blur.shape, thresh.shape, diff_img_3_channel.shape)
                concat_img = np.concatenate(
                    [blur, np.zeros_like(blur[:, :4]), thresh, np.zeros_like(thresh[:, :4]), diff_img_3_channel],
                    axis=1)
                plt.figure(figsize=(30 * 4, 30))
                plt.title('Thresh')
                plt.imshow(concat_img)
                plt.show()
                if pause_time is not None:
                    plt.ion()
                    plt.pause(pause_time)
                    plt.close('all')

            if self.save_img_dir is not None:
                blur = np.repeat(blur[:, :, None], 3, axis=-1)
                thresh = np.repeat(thresh[:, :, None], 3, axis=-1)
                print('blur.shape, thresh.shape, diff_img_3_channel.shape: ', blur.shape, thresh.shape,
                      diff_img_3_channel.shape)
                concat_img = np.concatenate(
                    [blur, np.zeros_like(blur[:, :4]), thresh, np.zeros_like(thresh[:, :4]), diff_img_3_channel],
                    axis=1)
                cv2.imwrite(os.path.join(self.save_img_dir, 'contour_{0:06d}.png'.format(self.step_index)), concat_img)

            center_list = sorted(center_list, key=lambda x: x[1])
            print(center_list)

            self.step_index = self.step_index + 1
            if self.agent_id == 0 and len(center_list) > 1:
                vehicle_0 = center_list[0]
                vehicle_1 = center_list[1]
                if np.sum(np.abs(vehicle_1[0] - vehicle_0[0]) + np.abs(
                        vehicle_1[1] - vehicle_0[1])) < self.distance_thres:
                    return (True, None)
                else:
                    return (False, None)
            else:
                return (False, None)
        elif self.collision_detection_mode == 'robot':
            print('2:agent ego_vehicle_location: ', ego_vehicle_location)
            # ego_vehicle_location: Location(x=132.030029, y=201.170212, z=0.182030)
            # local_ego_vehicle_location: (100.63402784916391, 47.810590826946736)
            # ego_vehicle_location:  Location(x=132.030029, y=201.170212, z=0.182030)
            local_ego_vehicle_location = ((ego_vehicle_location_offset[0] - ego_vehicle_location.x / 0.22),
                                          (ego_vehicle_location.y / 0.22 - ego_vehicle_location_offset[1]))  # [vertical, horizontal]
            # local_ego_vehicle_location:  (95.03753662109375, 291.1569118499756)
            print('2:agent local_ego_vehicle_location: ', local_ego_vehicle_location)
            t2no_img_wo_ego = copy.deepcopy(t2no_img)
            # plt.imshow(t2no_img)
            # plt.show()
            # plt.imshow(t2no_img_wo_ego)
            # plt.show()
            if self.vis and self.step_index > self.start_vis_frame_index:
                plt.title('T2NO')
                plt.imshow(t2no_img_wo_ego)
                plt.show()
                if pause_time is not None:
                    plt.ion()
                    plt.pause(pause_time)
                    plt.close('all')

            t2no_img_wo_ego[
                            int(local_ego_vehicle_location[0]-self.ego_car_size[0]):int(local_ego_vehicle_location[0]+self.ego_car_size[0]),
                            int(local_ego_vehicle_location[1]-self.ego_car_size[1]):int(local_ego_vehicle_location[1]+self.ego_car_size[1]),
                            ] = t2no_img.max()
            start_point = (int(local_ego_vehicle_location[1]-self.ego_car_size[1]), int(local_ego_vehicle_location[0]-self.ego_car_size[0]))
            end_point = (int(local_ego_vehicle_location[1]+self.ego_car_size[1]), int(local_ego_vehicle_location[0]+self.ego_car_size[0]))
            t2no_img_wo_ego_rect = cv2.rectangle(t2no_img_wo_ego.astype('uint8'), start_point, end_point, (0, 0, int(t2no_img_wo_ego.max())), 1)
            occupied_area = self.his_length - t2no_img_wo_ego[int(local_ego_vehicle_location[0]-self.forehead_distance[0]):int(local_ego_vehicle_location[0]+self.forehead_distance[0]),
                                              int(local_ego_vehicle_location[1]-self.forehead_distance[1]):int(local_ego_vehicle_location[1]+self.forehead_distance[1]),
                                              ]
            # occupied_area = t2no_img_wo_ego[int(local_ego_vehicle_location[0]-self.forehead_distance[0]):int(local_ego_vehicle_location[0]+self.forehead_distance[0]),
            #                                   int(local_ego_vehicle_location[1]-self.forehead_distance[1]):int(local_ego_vehicle_location[1]+self.forehead_distance[1]),
            #                                   ]
            # occupied = np.sum(occupied_area) > self.occupied_area_thres
            try:
                # print('t2no_img_wo_ego.shape, occupied_area.shape: ', t2no_img_wo_ego.shape, occupied_area.shape,
                #       't2no_img_wo_ego.min(): {}, t2no_img_wo_ego.max(): {}, occupied_area.min(): {}, occupied_area.max(): {}'
                #       .format(t2no_img_wo_ego.min(), t2no_img_wo_ego.max(), occupied_area.min(), occupied_area.max()))
                print('np.sum(abs(occupied_area)): ', np.sum(np.abs(occupied_area)))
                # t2no_img_wo_ego_vis = t2no_img_wo_ego

                occupied_area_vis = (occupied_area * 255. / (self.his_length + 1)).astype('uint8')
            except:
                occupied_area = np.zeros((10, 10))
                occupied_area_vis = (occupied_area * 255. / (self.his_length + 1)).astype('uint8')
            t2no_img_wo_ego_rect_vis = (t2no_img_wo_ego_rect * 255. / (self.his_length + 1)).astype('uint8')
            # print('t2no_img_wo_ego_rect_vis.min(): {}, t2no_img_wo_ego_rect_vis.max(): {}'
            #       .format(t2no_img_wo_ego_rect_vis.min(), t2no_img_wo_ego_rect_vis.max()), 255. / (self.his_length + 1))
            # plt.title('T2NO w/o Ego')
            # plt.imshow(t2no_img_wo_ego_rect_vis)
            # plt.show()
            if self.vis and self.step_index > self.start_vis_frame_index:
                plt.title('T2NO w/o Ego')
                plt.imshow(t2no_img_wo_ego_rect_vis)
                plt.show()
                if pause_time is not None:
                    plt.ion()
                    plt.pause(pause_time)
                    plt.close('all')

                plt.title('Occupied Area')
                plt.imshow(occupied_area_vis)
                plt.show()
                if pause_time is not None:
                    plt.ion()
                    plt.pause(pause_time)
                    plt.close('all')
            if self.save_img_dir is not None:
                # plt.imshow(t2no_img_wo_ego_rect_vis)
                # plt.show()
                cv2.imwrite(os.path.join(self.save_img_dir, 't2no_wo_ego_{0:06d}.png'.format(self.step_index)), t2no_img_wo_ego_rect_vis)
                # cv2.imwrite(os.path.join(self.save_img_dir, 'occupied_area_{0:06d}.png'.format(self.step_index)),
                #             occupied_area_vis)

            self.step_index = self.step_index + 1
            # grid_img = t2no_img > 0
            # print('t2no_img.max(), t2no_img.min(): ', t2no_img.max(), t2no_img.min())
            # print('metrics: ', self.metrics)
            # plt.imshow(t2no_img)
            # plt.show()
            grid_img = t2no_img # <= step_thres
            # plt.imshow(grid_img)
            # plt.show()
            if self.agent_id == 0:
                if np.sum(np.abs(occupied_area)) > self.occupied_area_thres:
                    return (True, t2no_img_wo_ego, grid_img)
                else:
                    return (False, t2no_img_wo_ego, grid_img)
            else:
                return (False, t2no_img_wo_ego, grid_img)
        # x: 229 y: 476
        # x: 498 y: 472
        # return (False, None)
