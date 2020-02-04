import random
import numpy as np
import robosuite as suite
from robosuite.wrappers import MyGymWrapper

import numpy as np
from PIL import Image
import subprocess

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    low = np.array([0.5, 0.15])
    high = np.array([0.7, 0.6])

    print('low: ', low)
    print('high: ', high)

    env = MyGymWrapper(
        suite.make(
            'BinPackPlace',
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=1,
        ),
        action_bound=(low, high)
    )

    # env.viewer.set_camera(camera_id=1)

    for i_episode in range(4):
        observation = env.reset()
        for i in range(20):

            for _ in range(200):
                env.render()

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                for _ in range(200):
                    env.render()
                break
