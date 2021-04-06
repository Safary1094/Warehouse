import numpy as np
import multiprocessing as mp
from Config import Config
from Agent_A2C import Agent


def worker(main_e, work_e, agent: Agent):
    main_e.close()
    while True:
        cmd, data = work_e.recv()
        if cmd == 'gae':
            agent.make_arrays()
            agent.set_last_vals(*data)
            agent.GAE()
            agent.reshape()
            work_e.send(0)

        elif cmd == 'train':
            agent.train()
            work_e.send(0)

        elif cmd == 'get_rew':
            agent.get_rews(data)
            work_e.send(0)

        elif cmd == 'step':
            act = agent.make_step(*data)
            work_e.send(act)

        elif cmd == 'save':
            agent.save()
            work_e.send(0)

        elif cmd == 'load':
            agent.load()
            work_e.send(0)


class AgnStack:
    def __init__(self, agents, conf: Config):

        ctx = mp.get_context('spawn')

        self.main_end = []
        self.work_end = []
        for _ in range(conf.lift_n + conf.move_n):
            p, c = mp.Pipe()
            self.main_end.append(p)
            self.work_end.append(c)

        self.ps = []
        for main_e, work_e, agn in zip(self.main_end, self.work_end, agents):
            p = ctx.Process(target=worker, args=(main_e, work_e, agn))
            self.ps.append(p)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.global_steps = 0

    def gae(self, obs):
        for m, o in zip(self.main_end, obs):
            m.send(('gae', o))
        results = [m.recv() for m in self.main_end]

    def step(self, obs, dones):
        a=1
        for m, o, d in zip(self.main_end, obs, dones):
            m.send(('step', (*o, d)))
        acts = [m.recv() for m in self.main_end]
        return np.stack(acts, axis=1)

    def train(self):
        for m in self.main_end:
            m.send(('train', None))
        results = [m.recv() for m in self.main_end]

    def get_rew(self, rews):
        for m, rew in zip(self.main_end, rews):
            m.send(('get_rew', rew))
        results = [m.recv() for m in self.main_end]

    def save(self):
        for m in self.main_end:
            m.send(('save', None))
        results = [m.recv() for m in self.main_end]

    def load(self):
        for m in self.main_end:
            m.send(('load', None))
        results = [m.recv() for m in self.main_end]
