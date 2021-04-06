import sys
import numpy as np
import pandas as pd

from EnvStack import EnvStack
from WH_env import WH_env
from Config import Config
from AgentStack import AgnStack
from Agent_A2C import Agent


class logger:
    def __init__(self):
        self.filename = 'log.txt'
        # self.f = open(self.filename, "a")

    def log(self, msg):
        self.f = open(self.filename, "a")
        self.f.write(str(msg) + '\n')
        self.f.close()


class Academy:
    def __init__(self, conf: Config):
        self.conf = conf
        self.logger = logger()

        self.env = EnvStack([WH_env(self.conf) for _ in range(self.conf.nenv)], self.conf)
        self.env.ini_simulation()

        self.obs = self.env.reset()
        self.dones = [[0] * self.conf.nenv] * (self.conf.lift_n + self.conf.move_n)

        cuda_list = ["cuda:0", "cuda:0", "cuda:1"]
        names_list = ['lifter'] * self.conf.lift_n + ['mover'] * self.conf.move_n
        self.agentStack = AgnStack([Agent(name, self.conf, cuda) for name, cuda in zip(names_list, cuda_list)], self.conf)
        self.logDF = None

        if self.conf.load_model:
            self.agentStack.load()

    def train(self, t_start):
        for t in range(0, self.conf.train_epochs + 1, self.conf.traj_len):
            infos = self.generate_mb()
            if not self.conf.inference:
                self.agentStack.train()
                if (t + self.conf.traj_len) % self.conf.save_every_steps == 0:
                    self.agentStack.save()
                    self.logDF.to_csv('log.csv')

            if (t + self.conf.traj_len) % 1000 == 0:
                self.make_log(infos, t + self.conf.traj_len)

    def generate_mb(self):
        for n in range(self.conf.traj_len):
            acts = self.agentStack.step(self.obs, self.dones)
            self.obs, rewards, self.dones, infos = self.env.step(acts)
            if not self.conf.inference:
                self.agentStack.get_rew(rewards)

        if not self.conf.inference:
            self.agentStack.gae(self.obs)
        return infos

    def make_log(self, infos, t):
        step = np.array([i.step for i in infos]).sum()
        left = np.array([i.motions[0] for i in infos]).sum()
        right = np.array([i.motions[1] for i in infos]).sum()
        reward = np.array([i.reward for i in infos]).sum()
        collision = np.array([i.collision for i in infos]).sum()
        boxes = np.array([i.boxes for i in infos]).sum()
        # print(step)
        log = pd.DataFrame({'left': [left / step],
                            'right': [right / step],
                            'collision': [collision / step],
                            'boxes': [boxes],
                            'reward': [reward / step]}, index=[t])

        self.logDF = log if self.logDF is None else self.logDF.append(log)

        self.logger.log(log.round(3))
        # self.logger.log(' ')
        print(self.logDF.tail(1).round(3))


if __name__ == '__main__':
    work_folder = sys.argv[1]

    config = Config(work_folder)
    academy = Academy(config)

    start_step = 0
    academy.train(start_step)
