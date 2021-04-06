import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from Config import *


def conv_block(in_f, out_f, *args, **kwargs):
    conv2d = nn.Conv2d(in_f, out_f, *args, **kwargs)
    # torch.nn.init.xavier_uniform_(conv2d.weight)
    return nn.Sequential(conv2d,
                         nn.BatchNorm2d(out_f),
                         nn.LeakyReLU())


def conv_pool_block(in_f, out_f, *args, **kwargs):
    conv2d = nn.Conv2d(in_f, out_f, *args, **kwargs)
    padd = nn.MaxPool2d(2)
    # torch.nn.init.xavier_uniform_(conv2d.weight)
    return nn.Sequential(conv2d,
                         padd,
                         nn.BatchNorm2d(out_f),
                         nn.LeakyReLU())


def dec_block(in_f, out_f):
    dense = nn.Linear(in_f, out_f)
    # torch.nn.init.xavier_uniform_(dense.weight)
    return nn.Sequential(dense,
                         nn.LeakyReLU())


class PolicyNet(nn.Module):
    def __init__(self, net_arch, act_space):
        super().__init__()
        pix_layers = net_arch['pixel']
        dense_layers = net_arch['dense']
        fin_layers = net_arch['fine']
        pad = net_arch['paddings']
        ker = net_arch['window']
        pixel_conv = [conv_block(in_f, out_f, kernel_size=w, padding=p) for in_f, out_f, p, w in
                      zip(pix_layers, pix_layers[1:], pad, ker)]
        fine_conv = [conv_pool_block(in_f, out_f, kernel_size=3, stride=2) for in_f, out_f in
                     zip(fin_layers, fin_layers[1:])]
        dense_block = [dec_block(in_f, out_f) for in_f, out_f in zip(dense_layers, dense_layers[1:])]
        self.dense = nn.Sequential(*dense_block)
        self.pixel = nn.Sequential(*pixel_conv)
        self.fine = nn.Sequential(*fine_conv)
        self.head = nn.Sequential(nn.Linear(dense_layers[-1], act_space * 2), nn.Tanh())
        self.act_space = act_space

    def forward(self, den, pix, fin) -> MultivariateNormal:
        pix = self.pixel(pix)
        pix = torch.flatten(pix, start_dim=1)

        fin = self.fine(fin)
        fin = torch.flatten(fin, start_dim=1)

        dense = self.dense(torch.cat([pix, den, fin], dim=1))
        head = self.head(dense)
        mean, log_std = torch.split(head, dim=len(head.shape) - 1, split_size_or_sections=self.act_space)
        cov_mat = torch.diag_embed(torch.exp(log_std))
        pd = MultivariateNormal(mean, cov_mat)
        return pd


class Policy:
    def __init__(self, name, conf: Config, cuda):
        self.conf = conf
        self.pol = PolicyNet(self.conf.architecture[name], self.conf.act_space[name])
        self.optimizer = torch.optim.Adam(self.pol.parameters(), lr=conf.lr)
        self.device = torch.device(cuda)
        self.pol.cuda(self.device)
        print(self.pol)

    def loss(self, dense, pixel, fine, old_act, old_neglogp, advantages):
        cliprange = 0.1
        pd = self.pol(dense, pixel, fine)
        neglogp = -pd.log_prob(old_act)

        entropy = pd.entropy()
        ratio = torch.exp(old_neglogp - neglogp)

        pg_loss_unclip = - advantages * ratio
        pg_loss_clip = - advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        pol_loss = torch.max(pg_loss_unclip, pg_loss_clip) + self.conf.ent_coef * entropy

        return pol_loss.mean()

    def step(self, dense, pixel, fine, inference):
        dense = torch.tensor(dense, dtype=torch.float32, device=self.device)
        pixel = torch.tensor(pixel, dtype=torch.float32, device=self.device)
        fine = torch.tensor(fine, dtype=torch.float32, device=self.device)

        pd = self.pol(dense, pixel, fine)
        if inference:
            act = pd.mean
        else:
            act = pd.sample()
        neglogp = -pd.log_prob(act)
        return act.cpu().detach().numpy(), neglogp.detach().cpu().numpy(), pd

    def train_pol(self, dense, pixel, fine, old_act, returns, old_vals, old_neglogp):
        advantages = returns - old_vals
        if self.conf.batch_size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dense = torch.tensor(dense, dtype=torch.float32, device=self.device)
        pixel = torch.tensor(pixel, dtype=torch.float32, device=self.device)
        fine = torch.tensor(fine, dtype=torch.float32, device=self.device)
        old_act = torch.tensor(old_act, dtype=torch.float32, device=self.device)
        old_neglogp = torch.tensor(old_neglogp, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        L = self.loss(dense, pixel, fine, old_act, old_neglogp, advantages)
        L.backward()
        self.optimizer.step()


class ValueNet(nn.Module):
    def __init__(self, net_arch):
        super().__init__()
        pix_layers = net_arch['pixel']
        dense_layers = net_arch['dense']
        pad = net_arch['paddings']
        ker = net_arch['window']
        pixel_conv = [conv_block(in_f, out_f, kernel_size=k, padding=p) for in_f, out_f, p, k in
                      zip(pix_layers, pix_layers[1:], pad, ker)]
        dense_block = [dec_block(in_f, out_f) for in_f, out_f in zip(dense_layers, dense_layers[1:])]

        self.pixel = nn.Sequential(*pixel_conv)
        self.dense = nn.Sequential(*dense_block)
        self.head = nn.Sequential(nn.Linear(dense_layers[-1], 1))

    def forward(self, den, pix, fin):
        pix = self.pixel(pix)
        pix = torch.flatten(pix, start_dim=1)
        dense = self.dense(torch.cat([pix, den], dim=1))
        head = self.head(dense)
        return head[:, 0]


class Value:
    def __init__(self, name, conf: Config, cuda):
        self.val = ValueNet(conf.architecture[name])
        self.optimizer = torch.optim.Adam(self.val.parameters(), lr=conf.lr)
        self.device = torch.device(cuda)
        self.val.cuda(self.device)
        print(self.val)

    def loss(self, dense, pixel, fine, returns, old_val):
        cliprange = 0.1
        val = self.val(dense, pixel, fine)
        val_pred_clip = old_val + torch.clamp(val - old_val, -cliprange, cliprange)

        value_loss_unclip = (val - returns).pow(2)
        value_loss_clip = (val_pred_clip - returns).pow(2)

        loss = (torch.min(value_loss_unclip, value_loss_clip)).mean()

        return loss

    def step(self, dense, pixel, fine):
        dense = torch.tensor(dense, dtype=torch.float32, device=self.device)
        pixel = torch.tensor(pixel, dtype=torch.float32, device=self.device)
        fine = torch.tensor(fine, dtype=torch.float32, device=self.device)
        val = self.val(dense, pixel, fine)

        return val.detach().cpu().numpy()

    def train_val(self, dense, pixel, fine, returns, old_vals):
        self.optimizer.zero_grad()
        dense = torch.tensor(dense, dtype=torch.float32, device=self.device)
        pixel = torch.tensor(pixel, dtype=torch.float32, device=self.device)
        fine = torch.tensor(fine, dtype=torch.float32, device=self.device)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_vals = torch.tensor(old_vals, dtype=torch.float32, device=self.device)
        L = self.loss(dense, pixel, fine, returns, old_vals)
        L.backward()
        self.optimizer.step()
