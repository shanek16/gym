import os

import numpy as np
import rendering
from gym import Env
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gym.utils import seeding
from numpy import arctan2, array, cos, pi, sin
from PIL import Image, ImageDraw, ImageFont


class Plain_cycle(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        r_max=40,
        r_min=0,
        dt=0.05,
        v=1.0,
        d=10.0,
        l=3,  # noqa
        m=3,
        n=2,
        r_c=3,
        d_min=4.5,
        max_step=3600,  # one circle 1200 time steps
    ):  # m: # of target n: # of uavs
        self.observation_space = Dict(
            {  # r, beta, theta
                # only r, alpha does not imply information of abs position of uav
                "uav1_state": Box(
                    low=np.float32([r_min, -pi, -pi]),
                    high=np.float32([r_max, pi, pi]),
                    dtype=np.float32,
                ),
                "uav2_state": Box(
                    low=np.float32([r_min, -pi, -pi]),
                    high=np.float32([r_max, pi, pi]),
                    dtype=np.float32,
                ),
                "target1_position": Box(
                    low=np.float32([r_min, -pi]),
                    high=np.float32([r_max, pi]),
                    dtype=np.float32,
                ),
                "target2_position": Box(
                    low=np.float32([r_min, -pi]),
                    high=np.float32([r_max, pi]),
                    dtype=np.float32,
                ),
                "target3_position": Box(
                    low=np.float32([r_min, -pi]),
                    high=np.float32([r_max, pi]),
                    dtype=np.float32,
                ),
                "battery": Box(
                    low=np.float32([0, 0]),
                    high=np.float32([3000, 3000]),
                    dtype=np.float32,
                ),
                "surveillance": MultiBinary(
                    array([3])
                ),  # 1 or 0 [Target1, Target2, Target3]
                "charge_station_occupancy": Discrete(
                    3
                ),  # 0:free, 1:uav1 docked, 2:uav2 docked
            }
        )
        self.dt = dt
        self.v = v
        self.vdt = v * dt
        self.d = d  # target distance
        self.l = l  # coverage gap: coverage: d-l ~ d+l # noqa
        self.m = m  # of targets
        self.n = n  # of uavs
        self.r_c = r_c  # charge station radius
        self.omega_max = v / d_min
        self.action_space = MultiDiscrete(
            [2, 21, 2, 21]
        )  # {0= charge, 1~22 ->-1/4.5~1/4.5}
        self.step_count = None
        self.uav1_state = None
        self.uav2_state = None
        self.target1_state = None
        self.target2_state = None
        self.target3_state = None
        self.battery = array([3000, 3000])  # 3rounds*1200steps/round
        self.charge_station_occupancy = None
        self.surveillance = None
        # for debugging
        self.uav1_in_charge_station = 0
        self.uav2_in_charge_station = 0
        self.uav1docked_time = 0
        self.uav2docked_time = 0
        self.font = ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20
        )
        self.num2str = {0: "charge", 1: "surveillance"}

        self.seed()
        self.max_step = max_step
        self.viewer = None
        self.SAVE_FRAMES_PATH = "rand_frames_plain_01/"
        self.episode_counter = 0
        self.frame_counter = 0
        self.save_frames = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(
        self,
        uav1_pose=None,
        uav2_pose=None,
        target1_pose=None,
        target2_pose=None,
        target3_pose=None,
    ):
        self.episode_counter += 1
        self.step_count = 0
        if self.save_frames:
            os.makedirs(
                os.path.join(self.SAVE_FRAMES_PATH, f"{self.episode_counter:03d}"),
                exist_ok=True,
            )
            self.frame_counter = 0
        if uav1_pose is None:
            uav1_r = self.np_random.uniform(0, 40)  # D=40
            uav1_beta = self.np_random.uniform(-pi, pi)
            self.uav1_state = array(
                (
                    uav1_r * cos(uav1_beta),
                    uav1_r * sin(uav1_beta),
                    self.np_random.uniform(-pi, pi),  # = theta
                )
            )
        elif uav1_pose is not None:
            self.uav1_state = uav1_pose

        if uav2_pose is None:
            uav2_r = self.np_random.uniform(0, 40)  # D=40
            uav2_beta = self.np_random.uniform(-pi, pi)
            self.uav2_state = array(
                (
                    uav2_r * cos(uav2_beta),
                    uav2_r * sin(uav2_beta),
                    self.np_random.uniform(-pi, pi),  # = theta
                )
            )
        elif uav2_pose is not None:
            self.uav2_state = uav2_pose

        if target1_pose is None:
            target1_r = self.np_random.uniform(0, 30)  # 0~ D-d
            target1_beta = self.np_random.uniform(-pi, pi)
            self.target1_state = array(
                (target1_r * cos(target1_beta), target1_r * sin(target1_beta))
            )
        elif target1_pose is not None:
            self.target1_state = target1_pose

        if target2_pose is None:
            target2_r = self.np_random.uniform(0, 30)  # 0~ D-d
            target2_beta = self.np_random.uniform(-pi, pi)
            self.target2_state = array(
                (target2_r * cos(target2_beta), target2_r * sin(target2_beta))
            )
        elif target2_pose is not None:
            self.target2_state = target2_pose

        if target3_pose is None:
            target3_r = self.np_random.uniform(0, 30)  # 0~ D-d
            target3_beta = self.np_random.uniform(-pi, pi)
            self.target3_state = array(
                (target3_r * cos(target3_beta), target3_r * sin(target3_beta))
            )
        elif target3_pose is not None:
            self.target3_state = target3_pose

        self.battery = array([3000, 3000])
        self.uav1_in_charge_station = 0
        self.uav2_in_charge_station = 0
        self.uav1docked_time = 0
        self.uav2docked_time = 0
        self.charge_station_occupancy = 0
        self.surveillance = array([0, 0, 0])
        return self.observation

    def uav1kinematics(self, action):
        index2action = np.linspace(-1.0 / 4.5, 1.0 / 4.5, 21, dtype=np.float32)
        dtheta = index2action[action] * self.dt
        _lambda = dtheta / 2
        if _lambda == 0.0:
            self.uav1_state[0] += self.vdt * cos(self.uav1_state[-1])
            self.uav1_state[1] += self.vdt * sin(self.uav1_state[-1])
        else:
            ds = self.vdt * sin(_lambda) / _lambda
            self.uav1_state[0] += ds * cos(self.uav1_state[-1] + _lambda)
            self.uav1_state[1] += ds * sin(self.uav1_state[-1] + _lambda)
            self.uav1_state[2] += dtheta
            self.uav1_state[2] = wrap(self.uav1_state[2])

    def uav2kinematics(self, action):
        index2action = np.linspace(-1.0 / 4.5, 1.0 / 4.5, 21, dtype=np.float32)
        dtheta = index2action[action] * self.dt
        _lambda = dtheta / 2
        if _lambda == 0.0:
            self.uav2_state[0] += self.vdt * cos(self.uav2_state[-1])
            self.uav2_state[1] += self.vdt * sin(self.uav2_state[-1])
        else:
            ds = self.vdt * sin(_lambda) / _lambda
            self.uav2_state[0] += ds * cos(self.uav2_state[-1] + _lambda)
            self.uav2_state[1] += ds * sin(self.uav2_state[-1] + _lambda)
            self.uav2_state[2] += dtheta
            self.uav2_state[2] = wrap(self.uav2_state[2])

    def step(self, action):
        terminal = False
        # uav1
        battery1, battery2 = self.battery
        if action[0] == 0:  # charge
            if (
                self.observation1[0] < self.r_c and self.uav2_in_charge_station is False
            ):  # uav1_in_charge_station
                # uav1 no move
                self.uav1_in_charge_station = 1
                if self.uav1docked_time == 0:  # landing
                    self.uav1docked_time += 1
                    # battery stays the same(docking time)
                else:  # uav1docked_time > 0
                    battery1 = min(battery1 + 10, 3000)
                    self.uav1docked_time += 1
            else:  # not able to land on charge station(too far/uav2 is in)
                self.uav1_in_charge_station = 0
                self.uav1docked_time = 0
                battery1 -= 1
                self.uav1kinematics(action[1])
        else:  # surveillance
            self.uav1_in_charge_station = 0
            self.uav1docked_time = 0
            battery1 -= 1
            self.uav1kinematics(action[1])

        # uav2
        if action[2] == 0:  # uav2 charge
            if (
                self.observation2[0] < self.r_c and self.uav1_in_charge_station is False
            ):  # uav2_in_charge_station
                # uav2 no move
                self.uav2_in_charge_station = 2
                if self.uav2docked_time == 0:  # landing
                    self.uav2docked_time += 1
                    # battery stays the same(docking time)
                else:  # uav2docked_time > 0
                    battery2 = min(battery2 + 10, 3000)
                    self.uav2docked_time += 1
            else:  # not able to land on charge station(too far/uav1 is in)
                self.uav2_in_charge_station = 0
                self.uav2docked_time = 0
                battery2 -= 1
                self.uav2kinematics(action[3])
        else:  # uav2 surveillance
            self.uav2_in_charge_station = 0
            self.uav2docked_time = 0
            battery2 -= 1
            self.uav2kinematics(action[3])
        self.charge_station_occupancy = max(
            self.uav1_in_charge_station, self.uav2_in_charge_station
        )
        self.surveillance = self.cal_surveillance(action)
        self.battery = array([battery1, battery2])
        # print('battery: ', self.battery)

        # reward -=1 or 2 ~charge state
        if self.charge_station_occupancy > 0:
            reward_charge = -0.5
        else:
            reward_charge = -1
        
        # reward ~ surveillance
        reward_surveil = 3 * L1(self.surveillance) / self.n  # 0~3

        # reward ~ -danger field gradient proportional to left battery
        # https://www.desmos.com/calculator/giqpnvhtaf
        r_1 = self.observation1[0]
        r_2 = self.observation2[0]
        battery_boarder = 1500
        # danger field for uav1
        if battery1 > battery_boarder:
            reward_battery1 = 0
        else:
            if action[0]==0:
                reward_battery1 = 0
            else:
                reward_battery1 = -0.000002*(2200 - battery1)*r_1**2

        # danger field for uav2
        if battery2 > battery_boarder:
            reward_battery2 = 0
        else:
            if action[1]==0:
                reward_battery2 = 0
            else:
                reward_battery2 = -0.000002*(2200 - battery2)*r_2**2  

        # penalize monopoly of charge station
        reward_monopoly1 = 0
        if battery1 > 2500:
            if action[0]==0:
                reward_monopoly1 = -1
        reward_monopoly2 = 0
        if battery2 > 2500:
            if action[2]==0:
                reward_monopoly2 = -1

        # cirtical penalty when either one of uav falls
        reward_fall = 0
        if min(self.battery) == 0:
            reward_fall = -3600 * 2  # - max_timestep*2
            terminal = True
        reward = reward_charge + reward_surveil + reward_battery1 + reward_battery2 + reward_monopoly1 + reward_monopoly2 + reward_fall


        if self.step_count > self.max_step:
            terminal = True

        if self.save_frames:
            if int(self.step_count) % 6 == 0:
                # print('in if')
                image = self.render(action, mode="rgb_array")
                path = os.path.join(
                    self.SAVE_FRAMES_PATH,
                    f"{self.episode_counter:03d}",
                    f"{self.frame_counter+1:04d}.bmp",
                )
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
                text_1 = "uav1_in_charge_station: {}".format(self.uav1_in_charge_station)
                text0 = "uav2_in_charge_station: {}".format(self.uav2_in_charge_station)
                text1 = "charge station occupancy: {}".format(self.charge_station_occupancy)
                text2 = "surveillance : {}".format(self.surveillance)
                text3 = "uav1docked_time: {}".format(self.uav1docked_time)
                text4 = "uav2docked_time: {}".format(self.uav2docked_time)
                text5 = "uav1 action: {}".format(self.num2str[action[0]])
                text6 = "uav2 action: {}".format(self.num2str[action[2]])
                text7 = "uav1 battery: {}".format(self.battery[0])
                text8 = "uav2 battery: {}".format(self.battery[1])
                text9 = "R_c: {}".format(reward_charge)
                text10 = "R_s: {}".format(reward_surveil)
                text11 = "R_b1: {}".format(reward_battery1)
                text12 = "R_b2: {}".format(reward_battery2)
                text13 = "R_m1: {}".format(reward_monopoly1)
                text14 = "R_m2: {}".format(reward_monopoly2)
                text15 = "R_f: {}".format(reward_fall)
                text16 = "Reward: {}".format(reward)
                draw.text((0, 0), text_1, color=(200, 200, 200), font=self.font)
                draw.text((0, 20), text0, color=(200, 200, 200), font=self.font)
                draw.text((0, 40), text1, color=(200, 200, 200), font=self.font)
                draw.text((0, 60), text2, color=(200, 200, 200), font=self.font)
                draw.text((0, 80), text3, color=(200, 200, 200), font=self.font)
                draw.text((0, 100), text4, color=(200, 200, 200), font=self.font)
                draw.text((0, 120), text5, color=(255, 255, 0), font=self.font)
                draw.text((0, 140), text6, color=(255, 255, 255), font=self.font)
                draw.text((770, 0), text7, color=(255, 255, 255), font=self.font)
                draw.text((770, 20), text8, color=(255, 255, 255), font=self.font)
                draw.text((770, 40), text9, color=(255, 255, 255), font=self.font)
                draw.text((770, 60), text10, color=(255, 255, 255), font=self.font)
                draw.text((770, 80), text11, color=(255, 255, 255), font=self.font)
                draw.text((770, 100), text12, color=(255, 255, 255), font=self.font)
                draw.text((770, 120), text13, color=(255, 255, 255), font=self.font)
                draw.text((770, 140), text14, color=(255, 255, 255), font=self.font)
                draw.text((770, 160), text15, color=(255, 255, 255), font=self.font)
                draw.text((770, 180), text16, color=(255, 255, 255), font=self.font)
                image.save(path)
                self.frame_counter += 1
        self.step_count += 1
        return self.observation, reward, terminal, {}

    def render(self, action, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 1000)
            bound = int(40 * 1.05)
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        # target1
        target1_x, target1_y = self.target1_state
        # draw donut
        outer_donut = self.viewer.draw_circle(
            radius=self.d + self.l, x=target1_x, y=target1_y, filled=True
        )
        if self.surveillance[0] == 1:
            outer_donut.set_color(0.6, 0.6, 1.0, 0.3)  # lighter
        else:
            outer_donut.set_color(0.3, 0.3, 0.9, 0.3)  # transparent blue
        inner_donut = self.viewer.draw_circle(
            radius=self.d - self.l, x=target1_x, y=target1_y, filled=True
        )
        inner_donut.set_color(0, 0, 0)  # erase inner part
        # target1 & circle
        circle = self.viewer.draw_circle(
            radius=self.d, x=target1_x, y=target1_y, filled=False
        )
        circle.set_color(1, 1, 1)
        # draw target
        target1 = self.viewer.draw_circle(
            radius=2, x=target1_x, y=target1_y, filled=True
        )
        target1.set_color(1, 0.6, 0)  # orange

        # target2
        target2_x, target2_y = self.target2_state
        # draw donut
        outer_donut = self.viewer.draw_circle(
            radius=self.d + self.l, x=target2_x, y=target2_y, filled=True
        )
        if self.surveillance[1] == 1:
            outer_donut.set_color(0.6, 0.6, 1.0, 0.3)  # lighter
        else:
            outer_donut.set_color(0.3, 0.3, 0.9, 0.3)  # transparent blue
        inner_donut = self.viewer.draw_circle(
            radius=self.d - self.l, x=target2_x, y=target2_y, filled=True
        )
        inner_donut.set_color(0, 0, 0)  # erase inner part
        # target2 & circle
        circle = self.viewer.draw_circle(
            radius=self.d, x=target2_x, y=target2_y, filled=False
        )
        circle.set_color(1, 1, 1)
        # draw target
        target2 = self.viewer.draw_circle(
            radius=2, x=target2_x, y=target2_y, filled=True
        )
        target2.set_color(1, 0.6, 0)  # orange

        # target3
        target3_x, target3_y = self.target3_state
        # draw donut
        outer_donut = self.viewer.draw_circle(
            radius=self.d + self.l, x=target3_x, y=target3_y, filled=True
        )
        if self.surveillance[2] == 1:
            outer_donut.set_color(0.6, 0.6, 1.0, 0.3)  # lighter
        else:
            outer_donut.set_color(0.3, 0.3, 0.9, 0.3)  # transparent blue
        inner_donut = self.viewer.draw_circle(
            radius=self.d - self.l, x=target3_x, y=target3_y, filled=True
        )
        inner_donut.set_color(0, 0, 0)  # erase inner part
        # target3 & circle
        circle = self.viewer.draw_circle(
            radius=self.d, x=target3_x, y=target3_y, filled=False
        )
        circle.set_color(1, 1, 1)
        # draw target
        target3 = self.viewer.draw_circle(
            radius=2, x=target3_x, y=target3_y, filled=True
        )
        target3.set_color(1, 0.6, 0)  # orange

        # charge station @ origin
        charge_station = self.viewer.draw_circle(radius=self.r_c, filled=True)
        if self.charge_station_occupancy == 0:
            if action[0] == 0:  # uav1 charge
                charge_station.set_color(1, 1, 0)  # yellow
            elif action[2] == 0:  # uav2 charge
                charge_station.set_color(1, 1, 1)  # white
            else:
                charge_station.set_color(0.1, 0.9, 0.1)  # green
        else:
            charge_station.set_color(0.9, 0.1, 0.1)  # red

        # uav1 (yellow)
        uav1_x, uav1_y, uav1_theta = self.uav1_state
        uav1_tf = rendering.Transform(translation=(uav1_x, uav1_y), rotation=uav1_theta)
        uav1_tri = self.viewer.draw_polygon([(-0.8, 0.8), (-0.8, -0.8), (1.6, 0)])
        uav1_tri.set_color(1, 1, 0)  # yellow
        uav1_tri.add_attr(uav1_tf)
        # uav2 (white)
        uav2_x, uav2_y, uav2_theta = self.uav2_state
        uav2_tf = rendering.Transform(translation=(uav2_x, uav2_y), rotation=uav2_theta)
        uav2_tri = self.viewer.draw_polygon([(-0.8, 0.8), (-0.8, -0.8), (1.6, 0)])
        uav2_tri.set_color(1, 1, 1)  # white
        uav2_tri.add_attr(uav2_tf)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    # relative position
    @property
    def observation1(self):
        x, y = self.uav1_state[:2]
        r = (x**2 + y**2) ** 0.5
        alpha = wrap(arctan2(y, x) - wrap(self.uav1_state[-1]) - pi)
        return array([r, alpha, self.uav1_state[2]])  # beta

    @property
    def observation2(self):
        x, y = self.uav2_state[:2]
        r = (x**2 + y**2) ** 0.5
        alpha = wrap(arctan2(y, x) - wrap(self.uav2_state[-1]) - pi)
        return array([r, alpha, self.uav2_state[2]])  # beta

    @property
    def target1_obs(self):
        x, y = self.target1_state[:2]
        r = (x**2 + y**2) ** 0.5
        beta = arctan2(y, x)
        return array([r, beta])  # beta

    @property
    def target2_obs(self):
        x, y = self.target2_state[:2]
        r = (x**2 + y**2) ** 0.5
        beta = arctan2(y, x)
        return array([r, beta])  # beta

    @property
    def target3_obs(self):
        x, y = self.target3_state[:2]
        r = (x**2 + y**2) ** 0.5
        beta = arctan2(y, x)
        return array([r, beta])  # beta

    # absolute position
    def rel_observation(self, uav, target):
        if uav == 1:
            uav_x, uav_y = self.uav1_state[:2]
            beta = self.uav1_state[2]
        else:
            uav_x, uav_y = self.uav2_state[:2]
            beta = self.uav2_state[2]

        if target == 1:
            target_x, target_y = self.target1_state
        elif target == 2:
            target_x, target_y = self.target2_state
        else:
            target_x, target_y = self.target3_state

        x = uav_x - target_x
        y = uav_y - target_y
        r = (x**2 + y**2) ** 0.5
        alpha = wrap(arctan2(y, x) - wrap(beta) - pi)
        return array([r, alpha])

    def cal_surveillance(self, action):
        # is any uav surveilling target 1?
        if (
            self.d - self.l < self.rel_observation(uav=1, target=1)[0]
            and self.rel_observation(uav=1, target=1)[0] < self.d + self.l
            and action[0] != 0 # intent is not charging
        ) or (
            self.d - self.l < self.rel_observation(uav=2, target=1)[0]
            and self.rel_observation(uav=2, target=1)[0] < self.d + self.l
            and action[2] != 0 # intent is not charging
        ):
            s1 = 1
        else:
            s1 = 0
        # is any uav surveilling target 2?
        if (
            self.d - self.l < self.rel_observation(uav=1, target=2)[0]
            and self.rel_observation(uav=1, target=2)[0] < self.d + self.l
            and action[0] != 0 # intent is not charging
        ) or (
            self.d - self.l < self.rel_observation(uav=2, target=2)[0]
            and self.rel_observation(uav=2, target=2)[0] < self.d + self.l
            and action[2] != 0 # intent is not charging
        ):
            s2 = 1
        else:
            s2 = 0
        # is any uav surveilling target 3?
        if (
            self.d - self.l < self.rel_observation(uav=1, target=3)[0]
            and self.rel_observation(uav=1, target=3)[0] < self.d + self.l
            and action[0] != 0 # intent is not charging
        ) or (
            self.d - self.l < self.rel_observation(uav=2, target=3)[0]
            and self.rel_observation(uav=2, target=3)[0] < self.d + self.l
            and action[2] != 0 # intent is not charging
        ):
            s3 = 1
        else:
            s3 = 0
        return array([s1, s2, s3])

    @property
    def observation(self):
        dictionary_obs = {
            "uav1_state": self.observation1,
            "uav2_state": self.observation2,
            "target1_position": self.target1_obs,
            "target2_position": self.target2_obs,
            "target3_position": self.target3_obs,
            "battery": self.battery,
            "surveillance": self.surveillance,
            "charge_station_occupancy": self.charge_station_occupancy,
        }
        return dictionary_obs


def wrap(theta):
    if theta > pi:
        theta -= 2 * pi
    elif theta < -pi:
        theta += 2 * pi
    return theta


def L1(x):
    return sum(abs(x))


if __name__ == "__main__":
    # Number of actions
    uav_env = Plain_cycle()
    action_sample = uav_env.action_space.sample()
    print("action_sample: ", action_sample)

    # Number of features
    state_sample = uav_env.observation_space.sample()
    print("state_sample: ", state_sample)

    print(uav_env.observation_space.spaces["uav1_state"])
    print(uav_env.reset()["uav1_state"])
