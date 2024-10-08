import os
import sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path)
desired_path = os.path.expanduser("~/Project/model_guard/uav_paper/Stochastic optimal control/uav_dp/gym")
sys.path.append(desired_path)
import numpy as np
import gym
from gym import Env
from gym.spaces import Box, Dict, Discrete # MultiBinary, MultiDiscrete
from typing import Optional
import rendering

from mdp import Actions, States
from numpy import arctan2, array, cos, pi, sin
from PIL import Image, ImageDraw, ImageFont
import time

CONTROL_FREQEUCNY = 1 # 50 # [hz]
dt = 1/CONTROL_FREQEUCNY # 0.02 [s]
render_freq = 1

def wrap(theta):
    if theta > pi:
        theta -= 2 * pi
    elif theta < -pi:
        theta += 2 * pi
    return theta

class UAV1Target1_real(Env):
    '''
    from verion 2, with real parameters from GAZEBO
    '''
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    dkc_env = gym.make('DKC_real_Unicycle')
    min_rt = 1000 # [m]
    class UAV:
        Q = 22_000 #[mAh] battery capacity
        C_rate = 2
        D_rate = 0.41 # 1/2.442(battery runtime)
        r_max = 5_000 # [m]
        def __init__(
            self,
            state,
            v=17, #[m/s]
            ):
            self.v = v
            self.battery = self.Q
            self.state = state
            self.charging = 0
        
        def copy(self):
            # Create a new UAV instance with the same attributes
            return UAV1Target1_real.UAV(state=self.state.copy(), v=self.v, battery=self.battery)
    
        def move(self, action):
            dtheta = action * dt
            _lambda = dtheta / 2
            if _lambda == 0.0:
                self.state[0] += self.v*dt * cos(self.state[-1])
                self.state[1] += self.v*dt * sin(self.state[-1])
            else:
                ds = self.v*dt * sin(_lambda) / _lambda
                self.state[0] += ds * cos(self.state[-1] + _lambda)
                self.state[1] += ds * sin(self.state[-1] + _lambda)
                self.state[2] += dtheta
                self.state[2] = wrap(self.state[2])
        @property
        def obs(self): # observation of uav relative to charging station
            x, y = self.state[:2]
            r = np.sqrt(x**2 + y**2)
            beta = arctan2(y, x)
                        # beta                  # theta
            alpha = wrap(beta - self.state[-1] - pi)
            return array([r, alpha, beta])  # beta

    class Target:
        max_age = 72*3600 # [s] 72 hours
        d=40 #[m] keeping distance > d_min
        epsilon=2 # [m] coverage gap(coverage: d-l ~ d+l), 5% of d
        def __init__(self, state, age=0, motion_type='static', sigma_rayleigh=0.5):
            self.state = state
            self.surveillance = None
            self.age = age
            self.motion_type = motion_type
            self.sigma_rayleigh = sigma_rayleigh

        def copy(self):
            return UAV1Target1_real.Target(state=self.state.copy(), age=self.age, motion_type=self.motion_type, sigma_rayleigh=self.sigma_rayleigh)

        def cal_age(self):
            if self.surveillance == 0:  # UAV is not surveilling
                self.age = min(self.max_age, self.age + dt)  # Change age
            else:
                self.age = 0

        def update_position(self):
            if self.motion_type == 'rayleigh':
                speed = np.random.rayleigh(self.sigma_rayleigh)
                angle = np.random.uniform(0, 2*np.pi)
                dx = speed * np.cos(angle) * dt
                dy = speed * np.sin(angle) * dt
                self.state += np.array([dx, dy])

        @property
        def obs(self):
            x, y = self.state
            r = np.sqrt(x**2 + y**2)
            beta = np.arctan2(y, x)
            return np.array([r, beta])  # r, beta

    def __init__(
        self,
        m=1, # of uavs
        n=1, # of targets
        r_c=10, #[m] ~ r_min in toc
        seed = None
    ):
        super().__init__()
        self.seed = seed
        self.observation_space = Dict(
            {  # r, alpha
                "uav1_target1": Box(
                    low=np.float32([0, -pi]),
                    high=np.float32([self.UAV.r_max, pi]),
                    dtype=np.float32,
                ),
                "uav1_charge_station": Box(
                    low=np.float32([0, -pi]),
                    high=np.float32([self.UAV.r_max, pi]),
                    dtype=np.float32,
                ),
                "battery": Box(
                    low=np.float32([0]),
                    high=np.float32([self.UAV.Q]),
                    dtype=np.float32,
                ),
                # "battery": Discrete(self.UAV.Q),
                "age": Discrete(self.Target.max_age), #changeage
            }
        )
        self.action_space = Discrete(2, seed = self.seed)  # 0: charge, 1: surveillance
        self.dt = dt
        self.discount = 0.999
        self.d = self.Target.d  # target distance
        self.epsilon = self.Target.epsilon  # coverage gap: coverage: d-l ~ d+l
        self.m = m  # of targets
        self.n = n  # of uavs
        self.r_c = r_c  # charge station radius
        self.step_count = None
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
        self.num2str = {0: "charge", 1: "target_1"}
        self.max_step = 72*3600*CONTROL_FREQEUCNY
        self.viewer = None
        self.SAVE_FRAMES_PATH = "../../../../visualized/1U1T"
        self.episode_counter = 0
        self.frame_counter = 0
        self.save_frames = False
        # self.print_q_init()

        # initialization for Dynamic Programming
        self.n_r = round(self.dkc_env.r_max/self.dkc_env.v*10)
        self.n_alpha = 360
        self.n_u = 2 #21

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        self.distance_keeping_result00 = np.load(current_file_path+ os.path.sep + "dkc_r5.0_rt0.04_2a_sig0_val_iter.npz")
        self.distance_keeping_straightened_policy00 = self.distance_keeping_result00["policy"] # .data
        self.time_optimal_straightened_policy00 = np.load(current_file_path+ os.path.sep + "lengthened_toc_r5.0_2a_sig0_val_iter.npy")
        self.states = States(
            np.linspace(0.0, self.dkc_env.r_max, self.n_r, dtype=np.float32),
            np.linspace(
                -np.pi,
                np.pi - np.pi / self.n_alpha,
                self.n_alpha,
                dtype=np.float32,
            ),
            cycles=[np.inf, np.pi * 2],
            n_alpha=self.n_alpha,
        )

        self.actions = Actions(
            np.linspace(-self.dkc_env.omega_max, self.dkc_env.omega_max, self.n_u, dtype=np.float32).reshape(
                (-1, 1)
            )
        )

    def reset(
        self,
        uav_pose=None,
        target_pose=None,
        battery=None,
        age=0,
        target_type = 'static',
        sigma_rayleigh=0.5,
        seed: Optional[int] = None,
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
        
        if target_pose is None:
            target1_r = np.random.uniform(self.min_rt, self.UAV.r_max-self.Target.d)  # 0~ D-d
            target1_beta = np.random.uniform(-pi, pi)
            target_state = array((target1_r * cos(target1_beta), target1_r * sin(target1_beta)))
        else:
            target_state = target_pose
        self.target1 = self.Target(state = target_state, age=age, motion_type=target_type, sigma_rayleigh=sigma_rayleigh)

        if uav_pose is None:
            # uav_theta = np.random.uniform(-pi, pi)
            # set theta aligned to beta, alpha=0
            uav_theta = self.target1.obs[1]
            uav_state = array((0, 0, uav_theta)
            )
        else:
            uav_state = uav_pose
        self.uav1 = self.UAV(state=uav_state)

        return self.dict_observation, {}

    def print_q_init(self): #TODO: real parameters not implemented.
        '''
        Initalization function for printing q value on frame for debugging purpose.
        (Check if discretization and 1u1t value table's state setting matches with current discretization settings.)
        '''
        self.n_alpha = 10
        # simple version
        # self.target_discretized_r_space = np.arange(0,81,20)
        # self.charge_discretized_r_space = np.arange(0,51,10)
        # self.target_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        # self.charge_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        # self.battery_space = np.arange(0, 3100,1000)

        # complex version
        self.target_discretized_r_space = np.array([0,4,6,8,10,12,14,16,20,30,40,60,80])
        self.charge_discretized_r_space = np.array([0,2,3,5,6,10,20,30,40])
        self.target_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        self.charge_alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float32)
        self.battery_space = np.concatenate([np.arange(0, 500, 100), np.arange(500, 3100, 500)])

        self.age_space = np.arange(0, 1001, 100) #changeage
        # self.UAV1Target1_result00 = np.load(f"/home/shane16/Project/model_guard/uav_paper/Stochastic optimal control/uav_dp/RESULTS/1U1T_s6_age1000:100_gamma_{self.discount}_dt_{self.dt}_{'val'}_iter.npz")
        self.UAV1Target1_result00 = np.load(os.getcwd() + f"/RESULTS/1U1T_s6_age1000:100_gamma_{self.discount}_dt_{self.dt}_{'val'}_iter.npz")
        # self.UAV1Target1_straightened_policy00 = self.UAV1Target1_result00["policy"]
        self.UAV1Target1_values00 = self.UAV1Target1_result00["values"]
        # print('shape of UAV1Target1_straightened_policy00: ', np.shape(self.UAV1Target1_straightened_policy00))
        # print('shape of UAV1Target1_values00: ', np.shape(self.UAV1Target1_values00))

        self.uav1target1_states = States(
            # uav1_target1
            self.target_discretized_r_space, # [0]
            self.target_alpha_space,         # [1]
            # uav1_charge
            self.charge_discretized_r_space, # [2]
            self.charge_alpha_space,         # [3]
            # battery_state
            self.battery_space,              # [4]
            self.age_space,                  # [5]
            cycles=[np.inf, np.pi*2, np.inf, np.pi*2, np.inf, np.inf],
            n_alpha=self.n_alpha,
        )

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
            action += p * self.actions[int(self.distance_keeping_straightened_policy00[s])]
            # print('action: ', action)
        return action

    def step(self, action):
        terminal = False
        truncated = False
        action = np.squeeze(action)
        # action clipping is done in dp already
        self.uav1.charging = 0
        self.target1.update_position()
        if self.uav1.battery <= 0: # UAV dead
            pass
        else: # UAV alive: can take action
            if action == 0:  # go to charge uav1
                if (self.uav1.obs[0] < self.r_c):
                    # uav1 no move
                    self.uav1.charging = 1
                    self.uav1.battery = min(self.UAV.Q, self.uav1.battery + self.UAV.C_rate*self.UAV.Q/3600/CONTROL_FREQEUCNY)
                else:  # not able to land on charge station(too far)
                    self.uav1.battery = max(0, self.uav1.battery - self.UAV.D_rate*self.UAV.Q/3600/CONTROL_FREQEUCNY)
                    w1_action = self.toc_get_action(self.uav1.obs[:2])
                    self.uav1.move(w1_action)
            else:  # surveil target1
                self.uav1.battery = max(0, self.uav1.battery - self.UAV.D_rate*self.UAV.Q/3600/CONTROL_FREQEUCNY)
                w1_action = self.dkc_get_action(self.rel_observation[:2])
                self.uav1.move(w1_action)
        self.cal_surveillance()
        self.target1.cal_age()
        reward = -self.target1.age
        if self.save_frames:
            self.print_q_value()
            # if int(self.step_count) % render_freq == 0:
            image = self.render(action, mode="rgb_array")
            path = os.path.join(
                self.SAVE_FRAMES_PATH,
                f"{self.episode_counter:03d}",
                f"{self.frame_counter+1:04d}.bmp",
            )
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            # left upper corner
            text0 = f"r_c: {self.uav1.obs[0]}, a_c: {self.uav1.obs[1]}"
            text1 = f"r_t: {self.target1.obs[0]}, a_t: {self.target1.obs[1]}"
            text2 = f"max_Q: {self.max_Q:.2f}"
            text3 = f"dQ: {self.max_Q - self.min_Q:.2f}"
            text4 = f"argmax(Q0,Q1): {self.argmax_Q}"
            text5 = "battery: {}".format(self.uav1.battery)
            text6 = "age: {}".format(self.target1.age)
            text7 = "Reward: {}".format(reward)

            draw.text((0, 0), text0, color=(200, 200, 200), font=self.font)
            draw.text((0, 20), text1, color=(200, 200, 200), font=self.font)
            draw.text((0, 40), text2, color=(255, 255, 0), font=self.font)
            draw.text((0, 60), text3, color=(200, 200, 200), font=self.font)
            draw.text((0, 80), text4, color=(200, 200, 200), font=self.font)
            draw.text((0, 100), text5, color=(255, 255, 255), font=self.font)
            draw.text((0, 120), text6, color=(255, 255, 255), font=self.font)
            draw.text((0, 140), text7, color=(255, 255, 255), font=self.font)         
            image.save(path)
            self.frame_counter += 1
        self.step_count += 1
        if self.step_count >= self.max_step:
            truncated = True
        return self.dict_observation, reward, terminal, truncated, {}

    def dry_cal_surveillance(self, uav1_copy, target1_copy, r_t):
        if uav1_copy.battery <= 0: # UAV dead
            target1_copy.surveillance = 0
        else: # UAV alive
            if (self.d - self.epsilon < r_t < self.d + self.epsilon
                and uav1_copy.charging != 1):
                target1_copy.surveillance = 1
            else:
                target1_copy.surveillance = 0
        return target1_copy.surveillance

    def dry_step(self, action, future, discount):
        # Copying relevant instance variables
        uav1_copy = self.uav1.copy()
        target1_copy = self.target1.copy()
        step_count_copy = self.step_count

        terminal = False
        truncated = False
        action = np.squeeze(action)
        reward = 0

        dry_dict_observation = self.dict_observation.copy()
        for i in range(future):
            if truncated:
                break
            target1_copy.update_position()
            # Logic for UAV1's battery and actions
            uav1_copy.charging = 0
            if uav1_copy.battery <= 0:  # UAV dead
                pass
            else:
                if action == 0:
                    if (dry_dict_observation['uav1_charge_station'][0] < self.r_c):
                        uav1_copy.charging = 1
                        uav1_copy.battery = min(self.UAV.Q, uav1_copy.battery + self.UAV.C_rate*self.UAV.Q/3600/CONTROL_FREQEUCNY)
                    else:
                        uav1_copy.battery = max(0, uav1_copy.battery - self.UAV.D_rate*self.UAV.Q/3600/CONTROL_FREQEUCNY)
                        w1_action = self.toc_get_action(dry_dict_observation['uav1_charge_station'][:2])
                        uav1_copy.move(w1_action)
                    
                else:
                    uav1_copy.battery = max(0, uav1_copy.battery - self.UAV.D_rate*self.UAV.Q/3600/CONTROL_FREQEUCNY)
                    w1_action = self.dkc_get_action(dry_dict_observation['uav1_target1'][:2])
                    uav1_copy.move(w1_action)
            uav_x, uav_y, theta = uav1_copy.state
            target_x, target_y = target1_copy.state
            x = uav_x - target_x
            y = uav_y - target_y
            r_t = np.sqrt(x**2 + y**2)
            beta_t = arctan2(y, x)
            alpha_t = wrap(beta_t - theta - pi)
            self.dry_cal_surveillance(uav1_copy, target1_copy, r_t)
            target1_copy.cal_age()

            step_count_copy += 1
            if step_count_copy >= self.max_step:
                truncated = True

            dry_dict_observation = { # is this state s_{t+10}?: Yes it is
                # r, alpha
                "uav1_target1": np.float32([r_t, alpha_t]),
                "uav1_charge_station": np.float32([uav1_copy.obs[0], uav1_copy.obs[1]]),
                "battery":  np.float32(uav1_copy.battery),
                "age": target1_copy.age,
                # "previous_action": action
            }
            reward += -target1_copy.age*discount**i
        return dry_dict_observation, reward, terminal, truncated, {}


    def render(self, action, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 1000)
            bound = int(self.UAV.r_max * 1.05)
            self.viewer.set_bounds(-bound, bound, -bound, bound)
        
        comm_rad = self.viewer.draw_circle(radius=self.UAV.r_max, x=0, y=0, filled=False)
        comm_rad.set_color(1, 1, 1)
        min_survelliance_r = self.viewer.draw_circle(radius=self.min_rt, x=0, y=0, filled=False)
        min_survelliance_r.set_color(1, 1, 0)

        # target1
        target1_x, target1_y = self.target1.state
        # draw donut
        outer_donut = self.viewer.draw_circle(
            radius=self.d + self.epsilon, x=target1_x, y=target1_y, filled=True
        )
        if self.target1.surveillance == 1:
            outer_donut.set_color(0.6, 0.6, 1.0, 0.3)  # lighter
        else:
            outer_donut.set_color(0.3, 0.3, 0.9, 0.3)  # transparent blue
        inner_donut = self.viewer.draw_circle(
            radius=self.d - self.epsilon, x=target1_x, y=target1_y, filled=True
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
        if self.uav1.charging == 1:
            charge_station.set_color(0.9, 0.1, 0.1)  # red
        else:
            if action == 0:
                charge_station.set_color(1, 1, 0)  # yellow
            else:
                charge_station.set_color(0.1, 0.9, 0.1)  # green            

        # uav1 (yellow)
        if self.uav1.battery <= 0: # UAV dead
            pass
        else:
            uav1_x, uav1_y, uav1_theta = self.uav1.state
            uav1_tf = rendering.Transform(translation=(uav1_x, uav1_y), rotation=uav1_theta)
            uav1_tri = self.viewer.draw_polygon([(-0.8, 0.8), (-0.8, -0.8), (1.6, 0)])
            uav1_tri.set_color(1, 1, 1)  # (1,1,0)yellow
            uav1_tri.add_attr(uav1_tf)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    @property
    def rel_observation(self): # of uav relative to target
        uav_x, uav_y, theta = self.uav1.state
        target_x, target_y = self.target1.state
        x = uav_x - target_x
        y = uav_y - target_y
        r = np.sqrt(x**2 + y**2)
        beta = arctan2(y, x)
        alpha = wrap(beta - theta - pi)
        return array([r, alpha, beta])

    def cal_surveillance(self):
        if self.uav1.battery <= 0:
            self.target1.surveillance = 0
        else: # UAV alive
            if self.d - self.epsilon < self.rel_observation[0] < self.d + self.epsilon:
                self.target1.surveillance = 1 # uav1 is surveilling target 1
            else:
                self.target1.surveillance = 0

    @property
    def dict_observation(self):
        dictionary_obs = {
            # r, alpha
            "uav1_target1": np.float32(
                [self.rel_observation[0],
                self.rel_observation[1]]
                ),
            "uav1_charge_station": np.float32([self.uav1.obs[0], self.uav1.obs[1]]),
            "battery":  np.float32(self.uav1.battery),
            "age": self.target1.age,
        }
        return dictionary_obs
    
    def print_q_value(self):
        '''
        1. Compute q value and print save it to variables.
        2. Use it to print q values on frames when save_frames=True.
        '''
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
        self.max_Q = max(self.Q0, self.Q1)
        self.min_Q = min(self.Q0, self.Q1)
        self.argmax_Q = np.argmax([self.Q0, self.Q1])
        # print(f'max_Q: {max_Q:.0f}', end=' | ')
        # print(f'argmax_Q: {self.argmax_Q}', end=' | ')
        # print(f'maxQ - Q: {self.max_Q - self.min_Q:.0f}', end=' | ')
        # print(f'Q0: {self.Q0:.2f}, Q1: {self.Q1:.2f}')


if __name__ == "__main__":
    # test return of dry_step == step
    ''' env = UAV1Target1()
    env.reset()
    state0, reward0, _, _, _ = env.dry_step(action=0)
    state, reward, _, _, _ = env.step(action=0)
    print(state0)
    print(state)'''
    
    uav_env = UAV1Target1_real()
    # action_sample = uav_env.action_space.sample()
    # print("action_sample: ", action_sample)

    # Number of features
    '''state_sample = uav_env.observation_space.sample()
    print("state_sample: ", state_sample)
    print('uav_env.observation_space:', uav_env.observation_space)
    print('uav_env.action_space.n: ', uav_env.action_space.n)'''
    
    # testing env: heuristic policy
    repitition = 10
    avg_reward = 0
    for i in range(repitition):
        step = 0
        truncated = False
        obs, _ = uav_env.reset(seed=i)
        bat = obs['battery']
        age = obs['age']
        total_reward = 0
        start = time.time()
        while truncated == False:
            step += 1
            r_c = obs['uav1_charge_station'][0]
            if bat > 0.66*uav_env.UAV.Q:
                action = 1
            elif bat > 0.33*uav_env.UAV.Q:
                # previous_action = obs['previous_action']
                if age == 0 or age > 0.8*uav_env.Target.max_age: # uav was surveilling
                # if previous_action:
                    action = 1
                else: # uav was charging
                    action = 0
            else:
                action = 0
            obs, reward, _, truncated, _ = uav_env.step(action)
            print(time.time()-start)
            start = time.time()
            total_reward += reward
            bat = obs['battery']
            age = obs['age']
            # print(f'step: {step} | battery: {bat} | reward: {reward}') #, end=' |')
            # print(f'action: {action}')#, end=' |')
            # uav_env.print_q_value()
            uav_env.render(action, mode='rgb_array')
        print(f'{i}: {total_reward}')   
        avg_reward += total_reward
    avg_reward /= repitition
    print(f'average reward: {avg_reward}')
    
    # testing env: alternating action
    '''obs, _ = uav_env.reset()
    step = 0
    while step < 5000:
        step += 1
        # action_sample = uav_env.action_space.sample()
        if step < 1000:
            action_sample = 0
        elif step < 2000:
            action_sample = 1
        elif step < 3000:
            action_sample = 0
        elif step < 4000:
            action_sample = 1
        else:
            action_sample = 0
        # action_sample = 1
        obs, reward, _, truncated, _ = uav_env.step(action_sample)
        bat = obs['battery']
        print(f'step: {step} | battery: {bat} | reward: {reward}')
        uav_env.render(action_sample)'''