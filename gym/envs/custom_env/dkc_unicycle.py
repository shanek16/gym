import os
import sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path)
desired_path = os.path.expanduser("~/Project/model_guard/uav_paper/Stochastic optimal control/uav_dp/gym")
sys.path.append(desired_path)
import warnings
import numpy as np
from gym import Env
from gym.spaces import Box
from gym.utils import seeding
from numpy import arctan2, array, cos, pi, sin
import rendering

warnings.filterwarnings("ignore")


class DKC_Unicycle(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        r_max=80,
        r_min=0.0,
        sigma=0.0,
        dt=0.05,
        v=1.0,
        d=10.0,
        d_min=4.5,
        k1=0.0181,
        max_step=2000, # is max_step=2000 sufficient for the uav(r=75) to reach the target? -> Yes it is. it takes less than 1800 steps.
    ):  # 0.07273
        self.viewer = None
        self.observation_space = Box(
            low=array([r_min, -pi]), high=array([r_max, pi]), dtype=np.float32
        )
        self.dt = dt
        self.v = v
        self.vdt = v * dt
        self.d = d
        self.d_min = d_min
        self.r_min = r_min
        self.omega_max = v / d_min
        self.action_space = Box(
            low=array([-self.omega_max]), high=array([self.omega_max]), dtype=np.float32
        )
        self.sigma = sigma
        self.k1 = k1
        self.max_step = max_step
        self.step_count = None
        self.state = None
        self.seed()
        self.tol = 1e-12

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, pose=None):
        self.step_count = 0
        if pose is None:
            r = self.np_random.uniform(
                self.observation_space.low[0],
                self.observation_space.high[0] - self.d_min,
            )
            theta = self.np_random.uniform(-pi, pi)
            self.state = array(
                (r * cos(theta), r * sin(theta), self.np_random.uniform(-pi, pi))
            )
        else:
            self.state = pose
        return self.observation

    def step(self, action):
        terminal = False
        truncated = False
        # clipping action
        if action[0] > self.omega_max:
            action[0] = self.omega_max
        elif action[0] < -self.omega_max:
            action[0] = -self.omega_max
        dtheta = action[0] * self.dt
        _lambda = dtheta / 2
        if _lambda == 0.0:
            self.state[0] += self.vdt * cos(self.state[-1])
            self.state[1] += self.vdt * sin(self.state[-1])
        else:
            ds = self.vdt * sin(_lambda) / _lambda
            self.state[0] += ds * cos(self.state[-1] + _lambda)
            self.state[1] += ds * sin(self.state[-1] + _lambda)
            self.state[2] += dtheta
            self.state[2] = wrap(self.state[2])
        obs = self.observation
        # terminal = obs[0] > self.observation_space.high[0]
        # terminal = obs[0] < self.observation_space.low[0]
        reward = self.k1 * (obs[0] - self.d) ** 2 + (-self.v * cos(obs[1])) ** 2
        reward = -reward
        if self.step_count > self.max_step:
            truncated = True
        self.step_count += 1
        # if self.step_count % 100 == 0:
        #     print(self.step_count)
        # is_done = terminal or truncated
        return obs, reward, terminal, truncated, {}

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 1000)
            bound = self.observation_space.high[0] * 1.05
            self.viewer.set_bounds(-bound, bound, -bound, bound)
        x, y, theta = self.state
        target = self.viewer.draw_circle(radius=self.r_min, filled=True)
        target.set_color(1, 0.6, 0)
        circle = self.viewer.draw_circle(radius=self.d, filled=False)
        circle.set_color(1,1,1)
        tf = rendering.Transform(translation=(x, y), rotation=theta)
        tri = self.viewer.draw_polygon([(-0.8, 0.8), (-0.8, -0.8), (1.6, 0)])
        tri.set_color(0.5, 0.5, 0.9)
        tri.add_attr(tf)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    @property
    def observation(self):
        x, y = self.state[:2] #+ self.sigma * self.np_random.randn(2)  # self.sigma=0 anyways
        r = (x**2 + y**2) ** 0.5
        alpha = wrap(arctan2(y, x) - wrap(self.state[-1]) - pi)
        return array([r, alpha])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def wrap(theta):
    if theta > pi:
        theta -= 2 * pi
    elif theta < -pi:
        theta += 2 * pi
    return theta

if __name__ == '__main__':
    uav_env = DKC_Unicycle()
    action_sample = uav_env.action_space.sample()
    print("action_sample: ", action_sample)

    # Number of features
    state_sample = uav_env.observation_space.sample()
    print("state_sample: ", state_sample)

    print('uav_env.observation_space:', uav_env.observation_space)
    
    step = 0
    uav_env.reset()
    while step < 5000:
        step += 1
        action_sample = uav_env.action_space.sample()
        uav_env.step(action_sample)
        uav_env.render(action_sample)