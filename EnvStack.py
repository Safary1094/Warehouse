import numpy as np
import multiprocessing as mp
from Config import Config


def worker(main_e, work_e, env):
    main_e.close()
    while True:
        cmd, data = work_e.recv()
        if cmd == 'ini_simulation':
            work_e.send(env.ini_simulation())
        elif cmd == 'reset':
            work_e.send(env.reset())
        elif cmd == 'step':
            state, reward, done, info = env.step(data)
            if done[0] > 0:
                state = env.reset()
            work_e.send((state, reward, done, info))


class EnvStack:
    def __init__(self, envs, conf: Config):
        self.conf = conf
        ctx = mp.get_context('spawn')

        self.main_end = []
        self.work_end = []
        for _ in range(self.conf.nenv):
            p, c = mp.Pipe()
            self.main_end.append(p)
            self.work_end.append(c)

        self.ps = []
        for main_e, work_e, env in zip(self.main_end, self.work_end, envs):
            p = ctx.Process(target=worker, args=(main_e, work_e, env))
            self.ps.append(p)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.global_steps = 0

    def reset(self):
        for m in self.main_end:
            m.send(('reset', None))
        obs = [m.recv() for m in self.main_end]
        return self.reorganize_obs(obs)

    # def plan(self):
    #     for m in self.main_end:
    #         m.send(('plan', None))
    #     results = [m.recv() for m in self.main_end]

    def step(self, actions):
        for m, a in zip(self.main_end, actions):
            m.send(('step', a))
        results = [m.recv() for m in self.main_end]
        obs, rews, dones, infos = zip(*results)
        return self.reorganize_obs(obs), list(zip(*rews)), list(zip(*dones)), infos

    def ini_simulation(self):
        for m in self.main_end:
            m.send(('ini_simulation', None))
        results = [m.recv() for m in self.main_end]

    def reorganize_obs(self, obs):
        obs = list(zip(*obs))
        obs = [np.array(o) for o in obs]
        state = []
        for i in range(0, (self.conf.move_n + self.conf.lift_n) * 6, 6):
            state.append(tuple(obs[i:i+6]))
        return state
