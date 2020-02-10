import os
import tensorflow as tf
import numpy as np
from gym.wrappers import TimeLimit
from baselines.ppo2 import ppo2
from baselines import logger
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.bench import Monitor
from baselines.common.vec_env.vec_monitor import VecEnvWrapper
from robosuite.environments.bin_pack_place import BinPackPlace
from gym import spaces

import random
import robosuite as suite
from robosuite.wrappers import MyGymWrapper
from PIL import Image

LOAD_PATH = 'results/baselines/ppo_mlp_2layer_0.001lr_256stpes_4async_more_obj/model.pth'

if __name__ == "__main__":

    low = np.array([0.5, 0.15])
    high = np.array([0.7, 0.6])
    obj_names = (['Milk'] * 2 + ['Bread'] * 2 + ['Cereal'] * 2 + ['Can'] * 2) * 2

    env = suite.make(
        'BinPackPlace',
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=1,
        obj_names=obj_names
    )

    env.viewer.set_camera(camera_id=0)

    num_envs = 4

    env = MyGymWrapper(env, (low, high), num_envs=num_envs)

    model = ppo2.learn(network='mlp', env=env,
                           total_timesteps=0, nsteps=16, save_interval=100, lr=1e-3,
                           num_layers=2, load_path=LOAD_PATH)

    n_episode = 10
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    for i in range(n_episode):
        obs = env.reset()
        obs = np.array([obs.tolist()] * num_envs)

        for j in range(100):
            for _ in range(200):
                env.render()

            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            # print('action: ', actions[0])
            obs, rew, done, _ = env.step(actions[0])
            obs = np.array([obs.tolist()] * num_envs)

            if done:
                for _ in range(200):
                    env.render()

                break
