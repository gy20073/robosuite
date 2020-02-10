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
import argparse
import robosuite as suite
from robosuite.wrappers import MyGymWrapper


def train(args, env):
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    network = args.network
    logger.configure()

    model = ppo2.learn(network=network, env=env,
                       total_timesteps=args.total_timesteps, nsteps=args.nsteps, save_interval=args.save_interval, lr=args.lr,
                       num_layers=args.num_layers)

    model.save(args.save_path)


if __name__ == "__main__":

    ## params
    parser = argparse.ArgumentParser(description='Baseline Training...')

    parser.add_argument('--out_dir', type=str, default='results/baselines')
    parser.add_argument('--alg', type=str, default='ppo')
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--control_freq', type=int, default=1)

    parser.add_argument('--total_timesteps', type=int, default=90000)
    parser.add_argument('--nsteps', type=int, default=128)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--network', type=str, default='mlp')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--debug', type=str, default='more_obj')


    args = parser.parse_args()

    ## const
    PATH = os.path.dirname(os.path.realpath(__file__))
    low = np.array([0.5, 0.15])
    high = np.array([0.7, 0.6])
    # obj_names = (['Milk'] * 2 + ['Bread'] * 2 + ['Cereal'] * 2 + ['Can'] * 2) * 2

    obj_names = ['Milk'] + ['Bread'] + ['Cereal'] + ['Can']

    ## make env
    # Notice how the environment is wrapped by the wrapper
    env = suite.make(
        'BinPackPlace',
        has_renderer=args.render,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=args.control_freq,
        obj_names=obj_names
    )


    info_dir = args.alg + '_' + args.network + '_' + str(args.num_layers) + 'layer_' +\
               str(args.lr) + 'lr_' + str(args.nsteps) + 'stpes_' + str(args.num_envs) + 'async_' + args.debug

    args.save_dir = os.path.join(PATH, args.out_dir, info_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_path = os.path.join(args.save_dir, 'model.pth')

    env = MyGymWrapper(env, (low, high), num_envs=args.num_envs)
    env = Monitor(env, args.save_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env)

    ## log
    print(args)

    train(args, env)
