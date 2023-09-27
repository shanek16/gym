import os
import sys
# current_file_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_file_path)
desired_path = os.path.expanduser("~/Project/model_guard/uav_paper/Stochastic optimal control/uav_dp/gym")
sys.path.append(desired_path)
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
            {  # r, alpha
                "uav1_target1": Box(
                    low=np.float32([r_min, -pi]),
                    high=np.float32([r_max, pi]),
                    dtype=np.float32,
                ),
                "uav1_charge_station": Box(
                    low=np.float32([r_min, -pi]),
                    high=np.float32([r_max, pi]),
                    dtype=np.float32,
                ),
                # "battery": Box(
                #     low=np.float32([0]),
                #     high=np.float32([3000]),
                #     dtype=np.float32,
                # ),
                "battery": Discrete(3000),
                "age": Discrete(10),
                "previous_action": Discrete(2),
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
        self.battery = 3000  # 3rounds*1200steps/round
        self.age = 0
        self.previous_action = 0
        self.surveillance = None
        # for debugging
        self.uav1_in_charge_station = 0
        # self.uav1docked_time = 0
        self.font = ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20
        )
        self.num2str = {0: "charge", 1: "target_1"}

        self.max_step = max_step
        self.viewer = None
        self.SAVE_FRAMES_PATH = "~/Project/model_guard/uav_paper/Stochastic optimal control/uav_dp/visualized/1uav1target" # example. save frames path is set at surveillance_PPO.py
        self.episode_counter = 0
        self.frame_counter = 0
        self.save_frames = False

        # initialization for Dynamic Programming
        self.n_r = 800
        self.n_alpha = 360
        self.n_u = 21

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        # self.distance_keeping_result00 = np.load(current_file_path+ os.path.sep + "v2_80_dkc_mdp_fp64.npz")
        self.distance_keeping_result00 = np.load(current_file_path+ os.path.sep + "rev_80_dkc_mdp_fp64.npz")
        self.distance_keeping_straightened_policy00 = self.distance_keeping_result00["policy.data"]
        # self.time_optimal_straightened_policy00 = np.load(current_file_path+ os.path.sep + "v2_terminal_40+40_toc_policy_fp64.npy")
        self.time_optimal_straightened_policy00 = np.load(current_file_path+ os.path.sep + "terminal_40+40_toc_policy_fp64.npy")

        self.states = States(
            np.linspace(0.0, 80.0, self.n_r, dtype=np.float32),
            np.linspace(
                -np.pi,
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
        self.discount = 0.995

    def reset(
        self,
        uav1_pose=None,
        target1_pose=None,
        battery=None,
        age=None,
        previous_action=None,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=self.seed) # 1
        np.random.seed(seed)
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
        else:
            self.uav1_state = uav1_pose

        if target1_pose is None:
            target1_r = np.random.uniform(0, 30)  # 0~ D-d
            target1_beta = np.random.uniform(-pi, pi)
            self.target1_state = array(
                (target1_r * cos(target1_beta), target1_r * sin(target1_beta))
            )
        else:
            self.target1_state = target1_pose

        if battery is None:
            self.battery = 3000
        else:
            self.battery = battery
        
        if age is None:
            self.age = 0
        else:
            self.age = age

        if previous_action is None:
            self.previous_action = 0
        else:
            self.previous_action = previous_action

        # self.uav1docked_time = 0
        
        return self.observation, {}

    def print_q_init(self):
        # complex version
        self.target_discretized_r_space = np.append(np.arange(0,20,3), np.arange(20,81,10))
        self.charge_discretized_r_space = np.concatenate([np.arange(0,10), np.arange(10,51,20)])
        self.target_alpha_space = np.linspace(-np.pi, np.pi - np.pi / 10, 10, dtype=np.float32)
        self.charge_alpha_space = np.linspace(-np.pi, np.pi - np.pi / 10, 10, dtype=np.float32)
        self.battery_space = np.concatenate([np.arange(0, 2000, 100), np.arange(2000, 3100, 500)])
        self.age_space = np.arange(0, 11, 3) #11
        self.previous_action_space = np.arange(2) # np.array([0, 1]),
        self.UAV1Target1_result00 = np.load(f"../../../../RESULTS/v1_1U1T_R_gamma_{self.discount}_val_iter.npz")
        self.UAV1Target1_straightened_policy00 = self.UAV1Target1_result00["policy"]
        self.UAV1Target1_values00 = self.UAV1Target1_result00["values"]
        # print('shape of UAV1Target1_straightened_policy00: ', np.shape(self.UAV1Target1_straightened_policy00))
        # print('shape of UAV1Target1_values00: ', np.shape(self.UAV1Target1_values00))

        self.uav1target1_states = States(
            # uav1_target1
            self.target_discretized_r_space, # [0]
            self.target_alpha_space,                # [1]
            # uav1_charge
            self.charge_discretized_r_space, # [2]
            self.charge_alpha_space,                # [3]
            # battery_state
            self.battery_space,              # [4]
            self.age_space,                  # [5]
            self.previous_action_space,
            cycles=[None, np.pi*2, None, np.pi*2, None, None, None],
        )

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
        if self.battery <= 0: # UAV dead
            self.uav1_in_charge_station = 0
            self.surveillance = 0
        else: # UAV alive: can take action
            if action == 0:  # go to charge uav1
                if (self.observation1[0] < self.r_c):  # uav1_in_charge_station
                    # uav1 no move
                    self.uav1_in_charge_station = 1
                    # if self.uav1docked_time == 0:  # landing
                    #     self.uav1docked_time += 1
                    #     # battery stays the same(docking time)
                    # else:  # uav1docked_time > 0
                    self.battery = min(self.battery + 10, 3000)
                    # self.uav1docked_time += 1
                else:  # not able to land on charge station(too far)
                    self.uav1_in_charge_station = 0
                    # self.uav1docked_time = 0
                    self.battery -= 1
                    w1_action = self.toc_get_action(self.observation1[:2])
                    self.uav1kinematics(w1_action)
            else:  # surveil target1
                self.uav1_in_charge_station = 0
                # self.uav1docked_time = 0
                self.battery -= 1
                w1_action = self.dkc_get_action(self.rel_observation[:2])
                self.uav1kinematics(w1_action)
            self.cal_surveillance()

        self.cal_age()
        # if self.battery == 0:
        #     reward = -3000
        # else:
        reward = -self.age
        self.previous_action = action

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
                # text1 = "uav1docked_time: {}".format(self.uav1docked_time)
                # print('action: ', action) 0 or 1
                # print('type of action: ', type(action)) # np.ndarray
                text2 = "uav1 action: {}".format(self.num2str[int(action)])
                text3 = "surveillance:{}".format(self.surveillance)
                text4 = "age: {}".format(self.age)
                # right uppper corner
                text5 = "uav1 battery: {}".format(self.battery)
                text6 = "r11: {0:0.0f}".format(abs(self.rel_observation[0]-10))
                text7 = "Reward: {}".format(reward)

                draw.text((0, 0), text0, color=(200, 200, 200), font=self.font)
                # draw.text((0, 20), text1, color=(200, 200, 200), font=self.font)
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

    def dry_uav1kinematics(self, action, uav1_state_copy):
        dtheta = action * self.dt
        _lambda = dtheta / 2
        if _lambda == 0.0:
            uav1_state_copy[0] += self.vdt * cos(uav1_state_copy[-1])
            uav1_state_copy[1] += self.vdt * sin(uav1_state_copy[-1])
        else:
            ds = self.vdt * sin(_lambda) / _lambda
            uav1_state_copy[0] += ds * cos(uav1_state_copy[-1] + _lambda)
            uav1_state_copy[1] += ds * sin(uav1_state_copy[-1] + _lambda)
            uav1_state_copy[2] += dtheta
            # Assuming wrap is a helper function, we need to handle it too
            # uav1_state_copy[2] = wrap(uav1_state_copy[2])
        return uav1_state_copy

    def dry_cal_surveillance(self, uav1_in_charge_station_copy, rel_observation_uav1_target1):
        surveillance_copy = 0
        if (self.d - self.l < rel_observation_uav1_target1[0] < self.d + self.l
            and uav1_in_charge_station_copy != 1):
            surveillance_copy = 1
        return surveillance_copy

    def dry_cal_age(self, surveillance_copy, age_copy):
        if surveillance_copy == 0:
            age_copy = min(10, age_copy + 1)
        else:
            age_copy = 0
        return age_copy

    def dry_step(self, action):
        # Copying relevant instance variables
        battery_copy = self.battery
        uav1_in_charge_station_copy = self.uav1_in_charge_station
        surveillance_copy = self.surveillance
        uav1_in_charge_station_copy = self.uav1_in_charge_station
        age_copy = self.age
        step_count_copy = self.step_count
        uav1_state_copy = self.uav1_state.copy()

        terminal = False
        truncated = False
        action = np.squeeze(action)

        # Logic for UAV1's battery and actions
        if battery_copy <= 0:  # UAV dead
            uav1_in_charge_station_copy = 0
            surveillance_copy = 0
        else:
            if action == 0:
                if (self.observation1[0] < self.r_c):
                    uav1_in_charge_station_copy = 1
                    battery_copy = min(battery_copy + 10, 3000)
                else:
                    uav1_in_charge_station_copy = 0
                    battery_copy -= 1
                    w1_action = self.toc_get_action(self.observation1[:2])
                    uav1_state_copy = self.dry_uav1kinematics(w1_action, uav1_state_copy)
                surveillance_copy = self.dry_cal_surveillance(uav1_in_charge_station_copy, 
                                                            self.rel_observation)
            else:
                uav1_in_charge_station_copy = 0
                battery_copy -= 1
                w1_action = self.dkc_get_action(self.rel_observation[:2])
                uav1_state_copy = self.dry_uav1kinematics(w1_action, uav1_state_copy)
                surveillance_copy = self.dry_cal_surveillance(uav1_in_charge_station_copy, 
                                                            self.rel_observation)

        age_copy = self.dry_cal_age(surveillance_copy, age_copy)
        # if battery_copy == 0:
        #     reward = -3000
        # else:
        reward = -age_copy

        step_count_copy += 1
        if step_count_copy >= self.max_step:
            truncated = True

        # Dry observation
        # relative r, alpha of uav1_target1
        uav_x, uav_y, theta = uav1_state_copy
        target_x, target_y = self.target1_state
        x = uav_x - target_x
        y = uav_y - target_y
        r_t = (x**2 + y**2) ** 0.5
        beta_t = arctan2(y, x)
        alpha_t = wrap(beta_t - wrap(theta) - pi)

        # relative r, alpha of uav1_charge_station
        r_c = (uav_x**2 + uav_y**2) ** 0.5
                    # beta                # theta
        alpha_c = wrap(arctan2(uav_y, uav_x) - wrap(theta) - pi)

        dry_observation = {
            # r, alpha
            "uav1_target1": np.float32([r_t, alpha_t]),
            "uav1_charge_station": np.float32([r_c, alpha_c]),
            "battery":  np.float32(battery_copy),
            "age": age_copy,
            "previous_action": action
        }
        return dry_observation, reward, terminal, truncated, {}


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
        if self.uav1_in_charge_station == 0:
            if action == 0:
                charge_station.set_color(1, 1, 0)  # yellow
            else:
                charge_station.set_color(0.1, 0.9, 0.1)  # green
        else:
            charge_station.set_color(0.9, 0.1, 0.1)  # red

        # uav1 (yellow)
        if self.battery <= 0: # UAV dead
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
    @property
    def rel_observation(self):
        uav_x, uav_y, theta = self.uav1_state
        target_x, target_y = self.target1_state

        x = uav_x - target_x
        y = uav_y - target_y
        r = (x**2 + y**2) ** 0.5
        beta = arctan2(y, x)
        alpha = wrap(beta - wrap(theta) - pi)
        return array([r, alpha, beta])

    def cal_surveillance(self):
        # is any uav surveilling target 1?
        if (
            self.d - self.l < self.rel_observation[0] < self.d + self.l
            # and action[0] != 0 # intent is not charging
            and self.uav1_in_charge_station != 1 # uav 1 is not charging(on the way to charge is ok)
        ):
            self.surveillance = 1 # uav1 is surveilling target 1
        else:
            self.surveillance = 0

    def cal_age(self):
        if self.surveillance == 0: # uav1 is not surveilling
            self.age = min(10, self.age + 1)
        else:
            self.age = 0

    @property
    def observation(self):
        dictionary_obs = {
            # r, alpha
            "uav1_target1": np.float32(
                [self.rel_observation[0],
                self.rel_observation[1]]
                ),
            "uav1_charge_station": np.float32([self.observation1[0], self.observation1[1]]),
            "battery":  np.float32(self.battery),
            "age": self.age,
            "previous_action": self.previous_action
        }
        return dictionary_obs
    
    def print_q_value(self):
        # dry_step with 1 env
        # 1) get reward for each action: 0 & 1
        state0, reward0, _, _, _ = self.dry_step(action=0)
        state1, reward1, _, _, _ = self.dry_step(action=1)

        # 2) get Value for each action
        S0, P0 = self.uav1target1_states.computeBarycentric(state0)
        value0 = 0
        for s0, p0 in zip(S0, P0):
            value0 += p0 * self.UAV1Target1_values00[s0]

        S1, P1 = self.uav1target1_states.computeBarycentric(state1)
        value1 = 0
        for s1, p1 in zip(S1, P1):
            value1 += p1 * self.UAV1Target1_values00[s1]

        self.Q0 = reward0+self.discount*value0
        self.Q1 = reward1+self.discount*value1
        max_Q = max(self.Q0, self.Q1)
        min_Q = min(self.Q0, self.Q1)
        print(f'max_Q: {max_Q:.0f}', end=' | ')
        print(f'maxQ - Q: {max_Q-min_Q:.0f}', end=' | ')
        print(f'Q0: {self.Q0:.2f}, Q1: {self.Q1:.2f}')

def wrap(theta):
    if theta > pi:
        theta -= 2 * pi
    elif theta < -pi:
        theta += 2 * pi
    return theta


def L1(x):
    return sum(abs(x))


if __name__ == "__main__":
    # test return of dry_step == step
    # env = UAV1Target1()
    # env.reset()
    # state0, reward0, _, _, _ = env.dry_step(action=0)
    # state, reward, _, _, _ = env.step(action=0)
    # print(state0)
    # print(state)
    
    uav_env = UAV1Target1()
    # action_sample = uav_env.action_space.sample()
    # print("action_sample: ", action_sample)

    # Number of features
    # state_sample = uav_env.observation_space.sample()
    # print("state_sample: ", state_sample)
    # print('uav_env.observation_space:', uav_env.observation_space)
    # print('uav_env.action_space.n: ', uav_env.action_space.n)
    
    repitition = 10
    avg_reward = 0
    uav_env.print_q_init()
    for i in range(repitition):
        step = 0
        truncated = False
        obs, _ = uav_env.reset(seed=i)
        bat = obs['battery']
        total_reward = 0
        while truncated == False:
            step += 1
            if bat > 2000:
                action = 1
            elif bat > 1000:
                if uav_env.uav1_in_charge_station:
                    action = 0
                else:
                    action = 1
            else:
                action = 0
            
            obs, reward, _, truncated, _ = uav_env.step(action)
            total_reward += reward
            bat = obs['battery']
            print(f'step: {step} | battery: {bat} | reward: {reward}', end=' |')
            uav_env.print_q_value()
            uav_env.render(action)
        print(f'{i}: {total_reward}')   
        avg_reward += total_reward
    avg_reward /= repitition
    print(f'average reward: {avg_reward}') 
    
    # testing env: alternating action
    # obs, _ = uav_env.reset()
    # step = 0
    # while step < 5000:
    #     step += 1
    #     # action_sample = uav_env.action_space.sample()
    #     if step < 1000:
    #         action_sample = 0
    #     elif step < 2000:
    #         action_sample = 1
    #     elif step < 3000:
    #         action_sample = 0
    #     elif step < 4000:
    #         action_sample = 1
    #     else:
    #         action_sample = 0
    #     # action_sample = 1
    #     obs, reward, _, truncated, _ = uav_env.step(action_sample)
    #     bat = obs['battery']
    #     print(f'step: {step} | battery: {bat} | reward: {reward}')
    #     uav_env.render(action_sample)