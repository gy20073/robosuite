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

    subprocess.call(['rm', '-rf', 'frames'])
    subprocess.call(['mkdir', '-p', 'frames'])
    subprocess.call(['mkdir', '-p', 'demo'])
    time_step_counter = 0

    env = MyGymWrapper(
        suite.make(
            'BinPackPlace',
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=1,
        ),
        action_bound=(low, high)
    )

    # env.viewer.set_camera(camera_id=0)

    for i_episode in range(4):
        observation = env.reset()
        for i in range(100):

            for _ in range(20):
                image_data = env.sim.render(width=200, height=200, camera_name='birdview')
                img = Image.fromarray(image_data, 'RGB')
                img.save('frames/frame-%.10d.png' % time_step_counter)
                time_step_counter += 1

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                for _ in range(20):
                    image_data = env.sim.render(width=200, height=200, camera_name='birdview')
                    img = Image.fromarray(image_data, 'RGB')
                    img.save('frames/frame-%.10d.png' % time_step_counter)
                    time_step_counter += 1
                break

    subprocess.call(
        ['ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p',
         'demo/demo.mp4'])

    subprocess.call(['rm', '-rf', 'frames'])
