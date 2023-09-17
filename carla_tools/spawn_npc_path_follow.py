#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import weakref
import argparse
import logging
import random
from queue import Queue
from queue import Empty
from functools import partial
# ======================================================================================================================
# -- Global variables. -------------------------------------------------------------------------------------
# ======================================================================================================================


def process_img(listener_i, image, sensor_queue):
    image.convert(cc.Raw)
    image.save_to_disk('_out_{}/{:08d}'.format(listener_i, image.frame_number))
    sensor_queue.put((image.frame, listener_i))

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=0,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=0,
        type=int)
    argparser.add_argument(
        '--gamma_correction',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--number_of_rgb_camera',
        default=4,
        type=int,
        help='number of RGB camera (default: 4)')
    argparser.add_argument(
        '--camera_height',
        default=13.,
        type=float,
        help='height of RGB camera (default: 15., 13.)')
    argparser.add_argument(
        '--save_time_steps',
        default=2000,
        type=int,
        help='number saved time steps (default: 5000)')
    argparser.add_argument(
        '--vehicle_color',
        default='fixed',
        type=str,
        help='vehicle color: random / fixed')

    args = argparser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    sensor_list = []

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.load_world('Town02')
    client.set_timeout(10.0)

    try:
        traffic_manager = client.get_trafficmanager(args.tm_port)
        world = client.get_world()

        synchronous_master = False

        if args.sync:
            settings = world.get_settings()
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.2  # 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        # weather = carla.WeatherParameters(
        #     cloudyness=80.0,
        #     precipitation=30.0,
        #     sun_altitude_angle=70.0)
        # ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, MidRainyNoon,
        # HardRainNoon, SoftRainNoon,
        # ClearSunset, CloudySunset, WetSunset, WetCloudySunset,
        # MidRainSunset, HardRainSunset, SoftRainSunset
        # weather = carla.WeatherParameters.ClearNoon
        # weather = carla.WeatherParameters.CloudyNoon
        # weather = carla.WeatherParameters.SoftRainSunset
        # weather = carla.WeatherParameters.HardRainSunset
        # weather = carla.WeatherParameters.MidRainSunset
        # weather = carla.WeatherParameters.WetCloudySunset
        # weather = carla.WeatherParameters.WetSunset
        # weather = carla.WeatherParameters.CloudySunset
        # weather = carla.WeatherParameters.ClearSunset
        # weather = carla.WeatherParameters.SoftRainNoon
        # weather = carla.WeatherParameters.HardRainNoon
        # weather = carla.WeatherParameters.MidRainyNoon
        # weather = carla.WeatherParameters.WetCloudyNoon
        # world.set_weather(weather)
        # print(world.get_weather())

        # global sensors_callback
        # sensors_callback = []
        blueprint_library = world.get_blueprint_library()
        all_spectators = world.get_actors().filter('traffic.traffic_light*')
        print('len(spectators): {}'.format(len(all_spectators)))
        spectators = []
        for each_spectator in all_spectators:
            print(each_spectator, each_spectator.id, each_spectator.get_transform())
            # Town02, run ./CarlaUE.sh first every time
            # if each_spectator.id in [87, 88, 91, 92]:   # [, 313, 314, 316, 317]:
            if each_spectator.id in [87, 88, 91, 92, 94, 105, 106, 108]:   # [, 313, 314, 316, 317]:
                spectators.append(each_spectator)
        assert args.number_of_rgb_camera < len(all_spectators), \
            'args.number_of_rgb_camera: {} > len(spectators): {}'.format(args.number_of_rgb_camera, len(all_spectators))
        # setup sensors
        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queue = Queue()
        bp = blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(600))
        bp.set_attribute('image_size_y', str(600))
        bp.set_attribute('sensor_tick', str(0.0333))
        # Sensor00
        spectator_transform = spectators[2].get_transform()
        spectator_transform_00 = spectators[2].get_transform()

        # create sensor
        sensor_transform = carla.Transform(carla.Location(x=spectator_transform.location.x,
                                                          y=spectator_transform.location.y,
                                                          z=args.camera_height),
                                           carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        sensor_00 = world.spawn_actor(bp, sensor_transform)

        # add callbacks
        sensor_00.listen(lambda data: process_img(listener_i=0, image=data, sensor_queue=sensor_queue))

        sensor_list.append(sensor_00)
        # Sensor01
        spectator_transform = spectators[1].get_transform()

        # create sensor
        sensor_transform = carla.Transform(carla.Location(x=spectator_transform_00.location.x,
                                                          y=spectator_transform.location.y,
                                                          z=args.camera_height),
                                           carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        sensor_01 = world.spawn_actor(bp, sensor_transform)

        # add callbacks
        sensor_01.listen(lambda data: process_img(listener_i=1, image=data, sensor_queue=sensor_queue))

        sensor_list.append(sensor_01)
        # Sensor02
        spectator_transform = spectators[3].get_transform()
        spectator_transform_02 = spectators[3].get_transform()
        print('spectator_transform.location: ', spectator_transform.location)

        # create sensor
        sensor_transform = carla.Transform(carla.Location(x=spectator_transform.location.x + 40, # "+" means the camera is moving up
                                                          y=spectator_transform.location.y,
                                                          z=args.camera_height),
                                           carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        sensor_02 = world.spawn_actor(bp, sensor_transform)

        # add callbacks
        sensor_02.listen(lambda data: process_img(listener_i=2, image=data, sensor_queue=sensor_queue))

        sensor_list.append(sensor_02)
        # Sensor03
        spectator_transform = spectators[0].get_transform()

        # create sensor
        sensor_transform = carla.Transform(carla.Location(x=spectator_transform_02.location.x + 40,
                                                          y=spectator_transform.location.y,
                                                          z=args.camera_height),
                                           carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        sensor_03 = world.spawn_actor(bp, sensor_transform)

        # add callbacks
        sensor_03.listen(lambda data: process_img(listener_i=3, image=data, sensor_queue=sensor_queue))

        sensor_list.append(sensor_03)
        # Sensor04
        spectator_transform = spectators[4].get_transform()
        spectator_transform_04 = spectators[4].get_transform()

        # create sensor
        sensor_transform = carla.Transform(carla.Location(x=spectator_transform.location.x,
                                                          y=spectator_transform.location.y,
                                                          z=args.camera_height),
                                           carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        sensor_04 = world.spawn_actor(bp, sensor_transform)

        # add callbacks
        sensor_04.listen(lambda data: process_img(listener_i=4, image=data, sensor_queue=sensor_queue))

        sensor_list.append(sensor_04)
        # Sensor05
        spectator_transform = spectators[5].get_transform()

        # create sensor
        sensor_transform = carla.Transform(carla.Location(x=spectator_transform_04.location.x,
                                                          y=spectator_transform.location.y,
                                                          z=args.camera_height),
                                           carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        sensor_05 = world.spawn_actor(bp, sensor_transform)

        # add callbacks
        sensor_05.listen(lambda data: process_img(listener_i=5, image=data, sensor_queue=sensor_queue))

        sensor_list.append(sensor_05)
        # Sensor06
        spectator_transform = spectators[6].get_transform()
        spectator_transform_06 = spectators[6].get_transform()

        # create sensor
        sensor_transform = carla.Transform(carla.Location(x=spectator_transform.location.x,
                                                          y=spectator_transform.location.y,
                                                          z=args.camera_height),
                                           carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        sensor_06 = world.spawn_actor(bp, sensor_transform)

        # add callbacks
        sensor_06.listen(lambda data: process_img(listener_i=6, image=data, sensor_queue=sensor_queue))

        sensor_list.append(sensor_06)
        # Sensor07
        spectator_transform = spectators[7].get_transform()

        # create sensor
        sensor_transform = carla.Transform(carla.Location(x=spectator_transform_06.location.x,
                                                          y=spectator_transform.location.y,
                                                          z=args.camera_height),
                                           carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        sensor_07 = world.spawn_actor(bp, sensor_transform)

        # add callbacks
        sensor_07.listen(lambda data: process_img(listener_i=7, image=data, sensor_queue=sensor_queue))

        sensor_list.append(sensor_07)

        # add a world.tick after seeing up the cameras. So that they are created before all the vehicles.
        # That way you more guarantee these actors are spawned in the world before trying to spawn all the other
        # actors likes cars and walkers. It's possible the cameras are being spawned in between other actors and
        # therefore starting there frame grabs at different times.
        if not args.sync or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        blueprints = world.get_blueprint_library().filter(args.filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        spawn_points = world.get_map().get_spawn_points()
        # print('len(spawn_points)', len(spawn_points))  # len(spawn_points) 101
        # Route 1
        # Create route 1 from the chosen spawn points
        # route_1_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # route_1 = []
        # for ind in route_1_indices:
        #     route_1.append(spawn_points[ind].location)
        #
        # # Route 2
        # # Create route 2 from the chosen spawn points
        # route_2_indices = [10, 11, 12, 13, 14, 15]
        # route_2 = []
        # for ind in route_2_indices:
        #     route_2.append(spawn_points[ind].location)
        # # while True:
        # #     world.tick()
        #
        # # Now let's print them in the map so we can see our routes
        # world.debug.draw_string(spawn_points[0].location, 'Spawn point 1', life_time=30, color=carla.Color(255, 0, 0))
        # world.debug.draw_string(spawn_points[0].location, 'Spawn point 2', life_time=30, color=carla.Color(0, 0, 255))
        #
        # for ind in route_1_indices:
        #     world.debug.draw_string(spawn_points[ind].location, str(ind), life_time=60, color=carla.Color(255, 0, 0))
        #
        # for ind in route_2_indices:
        #     world.debug.draw_string(spawn_points[ind].location, str(ind), life_time=60, color=carla.Color(0, 0, 255))

        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            # blueprint = random.choice(blueprints)
            blueprint = blueprints[0]
            if blueprint.has_attribute('color'):
                if args.vehicle_color == 'random':
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                else:
                    color = blueprint.get_attribute('color').recommended_values[0]
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                # driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                driver_id = blueprint.get_attribute('driver_id').recommended_values[0]
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True))
            # if n == 0:
            #     traffic_manager.set_path(vehicle, route_1)
            #     traffic_manager.set_path(vehicle, route_2)
            batch.append(vehicle)

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not args.sync or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(60.0)
        img_counter = 0
        while img_counter < args.save_time_steps:
            if args.sync and synchronous_master:
                world.tick()
                w_frame = world.get_snapshot().frame
                print('\nWorld\'s frame: {}'.format(w_frame))
                try:
                    for _ in range(len(sensor_list)):
                        s_frame = sensor_queue.get(True, 1.0)
                        print(' Frame: {} Sensor: {}'.format(s_frame[0], s_frame[1]))
                except Empty:
                    print(' Some of the sensor infomation is missed')
                img_counter = img_counter + 1
            else:
                world.wait_for_tick()

    finally:

        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d sensors' % len(sensor_list))
        for sensor in sensor_list:
            sensor.stop()
            sensor.destroy()

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
