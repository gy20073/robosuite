import numpy as np
import robosuite as suite
import time

if __name__ == "__main__":

    # get the list of all environments
    envs = sorted(suite.environments.ALL_ENVS)

    # print info and select an environment
    print("Welcome to Surreal Robotics Suite v{}!".format(suite.__version__))
    print(suite.__logo__)
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        print("[{}] {}".format(k, env))
    print()
    try:
        s = input(
            "Choose an environment to run "
            + "(enter a number from 0 to {}): ".format(len(envs) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(envs))
    except:
        print("Input is not valid. Use 0 by default.")
        k = 0

    # initialize the task
    env = suite.make(
        envs[k],
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=1,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # do visualization
    start_all = time.time()
    NIter = 60
    for i in range(1):
        #env.reset()
        #env.viewer.set_camera(camera_id=0)

        for j in range(NIter):
            time1 = time.time()
            action = np.random.randn(env.dof)
            action = np.random.rand(2)* 2
            obs, reward, done, _ = env.step([0.6, 0.375])
            time2 = time.time()
            print("step time ", time2-time1, " second")

            for _ in range(200):
                #spass
                env.render()
            #env.render()

            print("render time ", time.time()-time2, " second")

    tot_time =time.time()-start_all
    tot_steps = NIter* (1.0/0.002)
    print(tot_time, " is the total time for", tot_steps," mujoco steps. FPS is ", tot_steps / tot_time)
