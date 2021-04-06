from Model import Policy, Value
from Config import Config
import numpy as np
import torch


class Agent:
    def __init__(self, name, conf: Config, cuda):
        self.conf = conf
        self.name = name

        self.policy = Policy(name, conf, cuda)
        self.value = Value('critic', conf, cuda)

        self.acts = []
        self.vals = []
        self.neglogs = []
        self.dones = []
        self.rews = []
        self.rets = []

        self.last_vals = None

        self.act_den = []
        self.act_pix = []
        self.act_fin = []
        self.cri_den = []
        self.cri_pix = []
        self.cri_fin = []

    def clear_mb(self):
        self.acts = []
        self.vals = []
        self.neglogs = []
        self.dones = []
        self.rews = []
        self.rets = None

        self.last_vals = None

        self.act_den = []
        self.act_pix = []
        self.act_fin = []
        self.cri_den = []
        self.cri_pix = []
        self.cri_fin = []

    def make_step(self, act_obs_fin, act_obs_den, act_obs_pix, cri_obs_fin, cri_obs_den, cri_obs_pix, dones):
        act, neglogp, pd = self.policy.step(act_obs_den, act_obs_pix, act_obs_fin, self.conf.inference)
        val = self.value.step(cri_obs_den, cri_obs_pix, cri_obs_fin)

        self.act_den.append(act_obs_den)
        self.act_pix.append(act_obs_pix)
        self.act_fin.append(act_obs_fin)
        self.cri_den.append(cri_obs_den)
        self.cri_pix.append(cri_obs_pix)
        self.cri_fin.append(cri_obs_fin)
        self.acts.append(act)
        self.vals.append(val)
        self.neglogs.append(neglogp)
        self.dones.append(dones)

        return act

    def get_rews(self, rews):
        self.rews.append(rews)

    def make_arrays(self):
        self.act_pix = np.asarray(self.act_pix, dtype=np.float32)
        self.act_den = np.asarray(self.act_den, dtype=np.float32)
        self.act_fin = np.asarray(self.act_fin, dtype=np.float32)
        self.cri_pix = np.asarray(self.cri_pix, dtype=np.float32)
        self.cri_den = np.asarray(self.cri_den, dtype=np.float32)
        self.cri_fin = np.asarray(self.cri_fin, dtype=np.float32)

        self.acts = np.asarray(self.acts, dtype=np.float32)
        self.vals = np.asarray(self.vals, dtype=np.float32)
        self.dones = np.asarray(self.dones, dtype=np.float32)
        self.neglogs = np.asarray(self.neglogs, dtype=np.float32)
        self.rews = np.asarray(self.rews, dtype=np.float32)

    def set_last_vals(self, act_obs_fin, act_obs_den, act_obs_pix, cri_obs_fin, cri_obs_den, cri_obs_pix):
        self.last_vals = self.value.step(cri_obs_den, cri_obs_pix, cri_obs_fin)

    def train(self):
        indices = np.arange(self.conf.traj_len * self.conf.nenv)
        for _ in range(self.conf.opt_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.conf.batch_size):
                mb_inds = indices[start:start + self.conf.batch_size]
                slices = [arr[mb_inds] for arr in (self.act_den, self.act_pix, self.act_fin, self.acts, self.rets, self.vals, self.neglogs)]
                self.policy.train_pol(*slices)

            for start in range(0, len(indices), self.conf.batch_size):
                mb_inds = indices[start:start + self.conf.batch_size]
                slices = [arr[mb_inds] for arr in (self.cri_den, self.cri_pix, self.cri_fin, self.rets, self.vals)]
                self.value.train_val(*slices)

        self.clear_mb()

    def GAE(self):
        advantages = np.zeros_like(self.rews)

        lastgaelam = 0
        n_steps = len(self.dones)
        # From last step to first step
        for t in reversed(range(n_steps)):
            # If t == before last step
            if t == n_steps - 1:
                nextnonterminal = 1.0 - n_steps
                nextvalues = self.last_vals
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.vals[t + 1]

            # Delta = R(st) + gamma * V(t+1) * nextnonterminal  - V(st)
            delta = self.rews[t] + self.conf.gam * nextvalues * nextnonterminal - self.vals[t]

            # Advantage = delta + gamma *  λ * nextnonterminal  * lastgaelam
            advantages[t] = lastgaelam = delta + self.conf.gam * self.conf.lam * nextnonterminal * lastgaelam

        # Returns
        self.rets = advantages + self.vals

    def reshape(self):
        self.act_pix = self.sf01(self.act_pix)
        self.act_den = self.sf01(self.act_den)
        self.act_fin = self.sf01(self.act_fin)
        self.cri_pix = self.sf01(self.cri_pix)
        self.cri_den = self.sf01(self.cri_den)
        self.cri_fin = self.sf01(self.cri_fin)

        self.acts = self.sf01(self.acts)
        self.rets = self.sf01(self.rets)
        self.vals = self.sf01(self.vals)
        self.neglogs = self.sf01(self.neglogs)

    def sf01(self, arr):
        s = arr.shape
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

    def save(self):
        torch.save(self.policy.pol.state_dict(), self.conf.work_folder + '/' + self.name + '_pol')
        torch.save(self.value.val.state_dict(), self.conf.work_folder + '/' + self.name + '_val')

    def load(self):
        self.policy.pol.load_state_dict(torch.load(self.conf.work_folder + '/' + self.name + '_pol'))
        self.policy.pol.eval()
        self.value.val.load_state_dict(torch.load(self.conf.work_folder + '/' + self.name + '_val'))
        self.value.val.eval()
