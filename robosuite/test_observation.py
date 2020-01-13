import numpy as np
import robosuite as suite
import random


if __name__ == '__main__':

    env = suite.make(
        'BinPackPlace',
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )

    # env.reset()
    # env.viewer.set_camera(camera_id=0)

    objs = 'Milk0'
    pp = env.sim.model.get_joint_qpos_addr(objs)[0]
    sim_state = env.sim.get_state()

    print('bin_pos: ', env.bin_pos)

    m_pos = sim_state.qpos[pp:pp+3]
    print('not in bin: ', env.not_in_bin(m_pos))
    print('m_pos: ', m_pos)


    for i in range(200):
        env.render()

    x = random.random() / 2
    y = random.random() / 2

    m_pos[0] += random.random() / 5
    m_pos[1] += random.random() / 5

    env.teleport_object(objs, x=m_pos[0], y=m_pos[1])

    do_action = False

    obs = None
    for i in range(1000):
        action = np.random.randn(env.dof)
        obs, reward, done, _ = env.step(action)
        env.render()

    print('obs: ', obs)
