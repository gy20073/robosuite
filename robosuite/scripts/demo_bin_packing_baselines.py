import os
import tensorflow as tf
from gym.wrappers import TimeLimit
from baselines.ppo2 import ppo2
from baselines import logger
from baselines.common.tf_util import get_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.bench import Monitor
from baselines.common.vec_env.vec_monitor import VecEnvWrapper
from robosuite.environments.bin_pack_place import BinPackPlace

import random
import robosuite as suite
from robosuite.wrappers import GymWrapper

import warnings

warnings.filterwarnings('ignore')

PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = os.path.join(PATH, 'results', 'baselines', 'ppo')
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)


def train(env, save=False):
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    network='mlp'
    logger.configure()
    model = ppo2.learn(network=network, env=env, total_timesteps=500000, nsteps=1000)

    if save:
        model.save(SAVE_PATH)

## TODO: complete baselines
if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = suite.make(
            'BinPackPlace',
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=1,
        )

    env = GymWrapper(env)
    env = Monitor(env, SAVE_PATH, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)

    train(env, True)
