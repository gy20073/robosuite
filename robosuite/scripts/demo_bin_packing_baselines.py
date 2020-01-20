import os
import tensorflow as tf
import numpy as np
from gym.wrappers import TimeLimit
from baselines.ppo2 import ppo2
from baselines import logger
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.bench import Monitor
from baselines.common.vec_env.vec_monitor import VecEnvWrapper
from robosuite.environments.bin_pack_place import BinPackPlace
from gym import spaces

import random
import robosuite as suite
from robosuite.wrappers import MyGymWrapper

import warnings

warnings.filterwarnings('ignore')

PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = None
SAVE_PATH = None


def train(env, save=False):
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    network = 'mlp'
    logger.configure()
    # import ipdb
    # ipdb.set_trace()
    model = ppo2.learn(network=network, env=env, total_timesteps=50000000, nsteps=1000)

    if save:
        model.save(SAVE_PATH)


## TODO: Problems: action space bound useless
if __name__ == "__main__":

    # low = np.array([0.405, 0.135])
    # high = np.array([0.8, 0.625])
    low = np.array([0.5, 0.15])
    high = np.array([0.7, 0.6])

    # Notice how the environment is wrapped by the wrapper
    env = suite.make(
        'BinPackPlace',
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=1
    )

    num_envs = 8

    SAVE_DIR = os.path.join(PATH, 'results', 'baselines', 'ppo')
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    SAVE_PATH = os.path.join(SAVE_DIR, 'model.pth')

    env = MyGymWrapper(env, (low, high), num_envs=num_envs)
    env = Monitor(env, SAVE_DIR, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)

    train(env, True)
