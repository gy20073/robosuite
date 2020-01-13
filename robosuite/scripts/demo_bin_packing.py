import random
import robosuite as suite
from robosuite.wrappers import GymWrapper

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            'BinPackPlace',
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=1,
        )
    )

    for i_episode in range(20):
        observation = env.reset()
        objs = env.mujoco_objects.items()
        for obj_name, obj_mjcf in objs:
            reward = 0
            position = [random.random(), random.random()]
            env.teleport_object(obj_name, x=position[0], y=position[1])

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            for i in range(200):
                env.render()

            print("reward: ", reward)
