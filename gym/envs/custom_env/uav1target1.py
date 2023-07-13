import os
import sys
# current_file_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_file_path)
desired_path = os.path.expanduser("~/Project/model_guard/uav_paper/Stochastic optimal control/uav_dp/gym")
sys.path.append(desired_path)
# print(sys.path)
import numpy as np
from gym import Env
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from typing import Optional
import rendering


from mdp import Actions, States
from numpy import arctan2, array, cos, pi, sin
from PIL import Image, ImageDraw, ImageFont


class UAV1Target1(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        r_max=80,
        r_min=0,
        dt=0.05,
        v=1.0,
        d=10.0,
        l=3, # noqa
        m=1, # of targets
        n=1, # of uavs
        r_c=3,
        d_min=4.5,
        max_step=6000,
        seed = None # one circle 1200 time steps
    ):
        super().__init__()
        self.seed = seed
        self.observation_space = Dict(
            {  # r, cos(alpha)
                "uav1_target1": Box(
                    low=np.float32([r_min, -1]),
                    high=np.float32([r_max, 1]),
                    dtype=np.float32,
                ),
                "uav1_charge_station": Box(
                    low=np.float32([r_min, -1]),
                    high=np.float32([r_max, 1]),
                    dtype=np.float32,
                ),
                "battery": Box(
                    low=np.float32([0]),
                    high=np.float32([3000]),
                    dtype=np.float32,
                ),
                "age": Discrete(10)
            }
        )
        self.action_space = Discrete(2, seed = self.seed)  # 0: charge, 1: surveillance
        self.dt = dt
        self.v = v
        self.vdt = v * dt
        self.d = d  # target distance
        self.l = l  # coverage gap: coverage: d-l ~ d+l # noqa
        self.m = m  # of targets
        self.n = n  # of uavs
        self.r_c = r_c  # charge station radius
        self.omega_max = v / d_min
        self.step_count = None
        self.uav1_state = None
        self.target1_state = None
        self.battery = array([3000])  # 3rounds*1200steps/round
        self.age = 0
        self.charge_station_occupancy = None
        self.surveillance = None
        # for debugging
        self.uav1_in_charge_station = 0
        self.uav1docked_time = 0
        self.font = ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20
        )
        self.num2str = {0: "charge", 1: "target_1"}

        self.max_step = max_step
        self.viewer = None
        self.SAVE_FRAMES_PATH = "~/Project/model_guard/uav_paper/Stochastic optimal control/uav_dp/visualized/1uav1target" # example. save frames path is set at surveillance_PPO.py
        self.episode_counter = 0
        self.frame_counter = 0
        self.save_frames = True

        # initialization for Dynamic Programming
        self.n_r = 800
        self.n_alpha = 360
        self.n_u = 21

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        self.distance_keeping_result00 = np.load(current_file_path+ os.path.sep + "80_dkc_result_0.0.npz")
        self.time_optimal_result00 = np.load(current_file_path+ os.path.sep + "80_toc_result_0.0.npz")
        
        self.distance_keeping_straightened_policy00 = self.distance_keeping_result00[
            "policy"
        ]
        self.time_optimal_straightened_policy00 = self.time_optimal_result00["policy"]

        self.states = States(
            np.linspace(0.0, 80.0, self.n_r, dtype=np.float32),
            np.linspace(
                -np.pi + np.pi / self.n_alpha,
                np.pi - np.pi / self.n_alpha,
                self.n_alpha,
                dtype=np.float32,
            ),
            cycles=[None, np.pi * 2],
        )

        self.actions = Actions(
            np.linspace(-1.0 / 4.5, 1.0 / 4.5, self.n_u, dtype=np.float32).reshape(
                (-1, 1)
            )
        )

    def reset(
        self,
        uav1_pose=None,
        target1_pose=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=self.seed) # 1
        self.episode_counter += 1
        self.step_count = 0
        if self.save_frames:
            # print('save_frames_path: ', self.SAVE_FRAMES_PATH)
            os.makedirs(
                os.path.join(self.SAVE_FRAMES_PATH, f"{self.episode_counter:03d}"),
                exist_ok=True,
            )
            self.frame_counter = 0
        if uav1_pose is None:
            uav1_r = np.random.uniform(0, 40)  # D=40
            uav1_beta = np.random.uniform(-pi, pi)
            self.uav1_state = array(
                (
                    uav1_r * cos(uav1_beta),
                    uav1_r * sin(uav1_beta),
                    np.random.uniform(-pi, pi),  # = theta
                )
            )
        elif uav1_pose is not None:
            self.uav1_state = uav1_pose

        if target1_pose is None:
            target1_r = np.random.uniform(0, 30)  # 0~ D-d
            target1_beta = np.random.uniform(-pi, pi)
            self.target1_state = array(
                (target1_r * cos(target1_beta), target1_r * sin(target1_beta))
            )
        elif target1_pose is not None:
            self.target1_state = target1_pose

        self.battery = array([3000])
        self.age = 0
        self.uav1_in_charge_station = 0
        self.uav2_in_charge_station = 0
        self.uav1docked_time = 0
        self.uav2docked_time = 0
        self.charge_station_occupancy = 0
        self.surveillance = 0
        return self.observation, {}

    def uav1kinematics(self, action):
        dtheta = action * self.dt
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

    def toc_get_action(self, state):
        S, P = self.states.computeBarycentric(state)
        action = 0
        for s, p in zip(S, P):
            action += p * self.actions[int(self.time_optimal_straightened_policy00[s])]
        return action

    def dkc_get_action(self, state):
        S, P = self.states.computeBarycentric(state)
        action = 0
        for s, p in zip(S, P):
            action += (
                p * self.actions[int(self.distance_keeping_straightened_policy00[s])]
            )
        return action

    def step(self, action):
        terminal = False
        truncated = False
        action = np.squeeze(action)
        # action clipping is done in dp already
        # uav1
        battery1 = self.battery[0]
        self.surveillance = 0
        if battery1 <= 0: # UAV dead
            self.uav1_in_charge_station = 0
        else: # UAV alive: can take action
            if action == 0:  # go to charge uav1
                if (
                    self.observation1[0] < self.r_c and self.uav2_in_charge_station == False
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
                    w1_action = self.toc_get_action(self.observation1[:2])
                    self.uav1kinematics(w1_action)
            else:  # surveil target1
                self.uav1_in_charge_station = 0
                self.uav1docked_time = 0
                battery1 -= 1
                w1_action = self.dkc_get_action(self.rel_observation(uav=1, target=1)[:2])
                self.uav1kinematics(w1_action)
            self.cal_surveillance(action)
            self.battery = array([battery1])

        self.charge_station_occupancy = self.uav1_in_charge_station
        self.cal_age()
        reward = -self.age

        if self.save_frames:
            if int(self.step_count) % 6 == 0:
                image = self.render(action, mode="rgb_array")
                path = os.path.join(
                    self.SAVE_FRAMES_PATH,
                    f"{self.episode_counter:03d}",
                    f"{self.frame_counter+1:04d}.bmp",
                )
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
                # left upper corner
                text0 = "uav1_in_charge_station: {}".format(self.uav1_in_charge_station)
                text1 = "uav1docked_time: {}".format(self.uav1docked_time)
                # print('action: ', action) 0 or 1
                # print('type of action: ', type(action)) # np.ndarray
                text2 = "uav1 action: {}".format(self.num2str[int(action)])
                text3 = "surveillance:{}".format(self.surveillance)
                text4 = "age: {}".format(self.age)
                # right uppper corner
                text5 = "uav1 battery: {}".format(self.battery[0])
                text6 = "r11: {0:0.0f}".format(abs(self.rel_observation(uav=1, target=1)[0]-10))
                text7 = "Reward: {}".format(reward)

                draw.text((0, 0), text0, color=(200, 200, 200), font=self.font)
                draw.text((0, 20), text1, color=(200, 200, 200), font=self.font)
                draw.text((0, 40), text2, color=(255, 255, 0), font=self.font)
                draw.text((0, 60), text3, color=(200, 200, 200), font=self.font)
                draw.text((0, 80), text4, color=(200, 200, 200), font=self.font)
                # right uppper corner
                draw.text((770, 0), text5, color=(255, 255, 255), font=self.font)
                draw.text((770, 20), text6, color=(255, 255, 255), font=self.font)
                draw.text((750, 40), text7, color=(255, 255, 255), font=self.font)         
                image.save(path)
                self.frame_counter += 1
        self.step_count += 1
        if self.step_count >= self.max_step:
            truncated = True
        return self.observation, reward, terminal, truncated, {}

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
        if self.surveillance == 1:
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
        if action== 1:
            target1.set_color(1, 1, 0)  # yellow
        else:
            target1.set_color(1, 0.6, 0)  # orange

        # charge station @ origin
        charge_station = self.viewer.draw_circle(radius=self.r_c, filled=True)
        if self.charge_station_occupancy == 0:
            if action == 0:
                charge_station.set_color(1, 1, 0)  # yellow
            else:
                charge_station.set_color(0.1, 0.9, 0.1)  # green
        else:
            charge_station.set_color(0.9, 0.1, 0.1)  # red

        # uav1 (yellow)
        if self.battery[0] <= 0: # UAV dead
            pass
        else:
            uav1_x, uav1_y, uav1_theta = self.uav1_state
            uav1_tf = rendering.Transform(translation=(uav1_x, uav1_y), rotation=uav1_theta)
            uav1_tri = self.viewer.draw_polygon([(-0.8, 0.8), (-0.8, -0.8), (1.6, 0)])
            uav1_tri.set_color(1, 1, 0)  # yellow
            uav1_tri.add_attr(uav1_tf)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    # relative position
    @property
    def observation1(self):
        x, y = self.uav1_state[:2]
        r = (x**2 + y**2) ** 0.5
                    # beta                  # theta
        alpha = wrap(arctan2(y, x) - wrap(self.uav1_state[-1]) - pi)
        beta = arctan2(y, x)
        return array([r, alpha, beta])  # beta

    @property
    def target1_obs(self):
        x, y = self.target1_state[:2]
        r = (x**2 + y**2) ** 0.5
        beta = arctan2(y, x)
        return array([r, beta])  # beta

    # absolute position
    def rel_observation(self, uav, target):
        if uav == 1:
            uav_x, uav_y, theta = self.uav1_state

        if target == 1:
            target_x, target_y = self.target1_state

        x = uav_x - target_x
        y = uav_y - target_y
        r = (x**2 + y**2) ** 0.5
        beta = arctan2(y, x)
        alpha = wrap(beta - wrap(theta) - pi)
        return array([r, alpha, beta])

    def cal_surveillance(self, action):
        # is any uav surveilling target 1?
        if (
            self.d - self.l < self.rel_observation(uav=1, target=1)[0] < self.d + self.l
            # and action[0] != 0 # intent is not charging
            and self.charge_station_occupancy != 1 # uav 1 is not charging(on the way to charge is ok)
        ):
            self.surveillance = 1 # uav1 is surveilling target 1

    def cal_age(self):
        if self.surveillance == 0: # uav1 is not surveilling
            self.age = min(10, self.age + 1)
        else:
            self.age = 0

    @property
    def observation(self):
        dictionary_obs = {
            # r, cos(alpha)
            "uav1_target1": np.float32(
                [self.rel_observation(uav=1, target=1)[0],
                cos(self.rel_observation(uav=1, target=1)[1])]
                ),
            "uav1_charge_station": np.float32([self.observation1[0], cos(self.observation1[1])]),
            "battery":  np.float32(self.battery),
            "age": self.age,
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
    uav_env = UAV1Target1()
    action_sample = uav_env.action_space.sample()
    print("action_sample: ", action_sample)

    # Number of features
    state_sample = uav_env.observation_space.sample()
    print("state_sample: ", state_sample)

    # print(uav_env.observation_space.spaces["uav1_state"])
    # print(uav_env.reset()["uav1_state"])

    print('uav_env.observation_space:', uav_env.observation_space)
    print('uav_env.action_space.n: ', uav_env.action_space.n)
    
    step = 0
    uav_env.reset()
    while step < 5000:
        step += 1
        # action_sample = uav_env.action_space.sample()
        # if step < 1000:
        #     action_sample = 0
        # elif step < 2000:
        #     action_sample = 1
        # elif step < 3000:
        #     action_sample = 0
        # elif step < 4000:
        #     action_sample = 1
        # else:
        #     action_sample = 0
        action_sample = 1
        uav_env.step(action_sample)
        uav_env.render(action_sample)