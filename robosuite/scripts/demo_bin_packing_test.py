import random
import numpy as np
import robosuite as suite
from robosuite.wrappers import MyGymWrapper

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    low = np.array([0, 0])
    high = np.array([0.5, 0.5])
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

    import ipdb
    ipdb.set_trace()
