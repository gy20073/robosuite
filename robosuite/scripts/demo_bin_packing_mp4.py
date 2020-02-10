import random
import numpy as np
import robosuite as suite
from robosuite.wrappers import MyGymWrapper
from baselines.ppo2 import ppo2

import os
import numpy as np
from PIL import Image
import subprocess

LOAD_PATH = '/home/yeweirui/checkpoint/openai-2020-02-06-04-07-39-814380/checkpoints/00900'

DEMO_PATH = 'demo'

if not os.path.exists(DEMO_PATH):
    os.makedirs(DEMO_PATH)

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    low = np.array([0.5, 0.15])
    high = np.array([0.7, 0.6])
    # obj_names = (['Milk'] * 2 + ['Bread'] * 2 + ['Cereal'] * 2 + ['Can'] * 2) * 2

    obj_names = ['Milk'] + ['Bread'] + ['Cereal'] + ['Can']

    subprocess.call(['rm', '-rf', 'frames'])
    subprocess.call(['mkdir', '-p', 'frames'])
    subprocess.call(['mkdir', '-p', 'demo'])
    time_step_counter = 0

    num_envs = 4

    env = MyGymWrapper(
        suite.make(
            'BinPackPlace',
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=1,
            obj_names=obj_names
        ),
        action_bound=(low, high),
        num_envs=num_envs
    )

    model = ppo2.learn(network='mlp', env=env,
                       total_timesteps=0, nsteps=16, save_interval=100, lr=1e-3,
                       num_layers=2, load_path=LOAD_PATH)

    n_episode = 10
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))


    for i_episode in range(4):
        obs = env.reset()
        obs = np.array([obs.tolist()] * num_envs)

        for i in range(100):

            for _ in range(20):
                image_data = env.sim.render(width=640, height=480, camera_name='birdview')
                img = Image.fromarray(image_data, 'RGB')
                img.save('frames/frame-%.10d.png' % time_step_counter)
                time_step_counter += 1

            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            # print('action: ', actions[0])
            obs, rew, done, _ = env.step(actions[0])
            obs = np.array([obs.tolist()] * num_envs)

            if done:
                for _ in range(20):
                    image_data = env.sim.render(width=640, height=480, camera_name='birdview')
                    img = Image.fromarray(image_data, 'RGB')
                    img.save('frames/frame-%.10d.png' % time_step_counter)
                    time_step_counter += 1
                break

    subprocess.call(
        ['ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p', '-s', '640x480',
         DEMO_PATH + '/demo.mp4'])

    subprocess.call(['rm', '-rf', 'frames'])
