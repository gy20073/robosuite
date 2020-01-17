import random
import numpy as np
import robosuite as suite
from robosuite.wrappers import MyGymWrapper

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    low = np.array([-0.1, -0.1])
    high = np.array([0.8,0.8])
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

    env.viewer.set_camera(camera_id=0)

    for i_episode in range(20):
        observation = env.reset()
        objs = env.mujoco_objects.items()
        for i in range(100):

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            for _ in range(1000):
                env.render()

            print("reward: ", reward)

            if done:
                print('Done!')
                break
