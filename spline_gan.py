import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from scipy.stats import kendalltau, norm

import pandas as pd
import numpy as np

import math


# Models
class Generator(nn.Module):
    # General covariance structure
    def __init__(self, p):
        super(Generator, self).__init__()
        self.p = p
        self.loc = nn.Parameter(torch.zeros(self.p))
        self.precision = nn.Linear(p, p, bias=False)

    def forward(self, x):
        # g(x) = Cov^{-1/2}(x - mu)
        return self.precision(x + self.loc)


class Discriminator(nn.Module):
    # linear truncated spline basis with pairwise interactions
    def __init__(self, p, knots):
        super(Discriminator, self).__init__()
        self.p = p
        self.knots = knots  # k-dim list of floats or tensor floats
        self.aug_linear_dim = (1 + len(self.knots)) * p
        self.acf = nn.ReLU()
        self.outer = nn.Bilinear(self.aug_linear_dim,
                                 self.aug_linear_dim,
                                 1, bias=True)
        self.linear = nn.Linear(self.aug_linear_dim, 1, bias=False)

    def forward(self, x):
        spline_basis = self.acf(torch.cat([x - knot for knot in self.knots], dim=1))
        aug = torch.cat([x, spline_basis], dim=1)
        return self.linear(aug) + self.outer(aug, aug)


# Robust spline GAN
class SplineGAN:
    def __init__(self, configs):
        self.configs = configs
        self.device = self.configs.device

        # Set universal random seed for reproducibility
        torch.manual_seed(self.configs.seed)

        # Set default dtype for GPU
        torch.set_default_dtype(torch.float32)

        # Generate real data X under config model setting
        self.real, self.mu, self.Sigma = self.sampler_real()
        self.precision = torch.linalg.inv(self.Sigma)

        # Prepare mini-batch data loader
        self.data_loader = DataLoader(self.real, batch_size=self.configs.batch_size, shuffle=True)

        # Setup networks and loss
        self.netG = Generator(self.configs.p).to(self.device)
        self.netD = Discriminator(self.configs.p, knots=[-1, 0, 1]).to(self.device)

        self.loss_d, self.loss_g = self.loss_select(self.configs.loss)

        # Setup optimizers
        self.opt = optim.Adam

        self.optimizerD = self.opt(self.netD.parameters(), lr=self.configs.lr_d)
        if self.configs.no_loc:
            self.optimizerG = self.opt([{'params': self.netG.precision.weight, 'lr': self.configs.lr_g},
                                        {'params': self.netG.loc, 'lr': 0}],
                                       lr=self.configs.lr_g)
        else:
            self.optimizerG = self.opt([{'params': self.netG.precision.weight, 'lr': self.configs.lr_g},
                                        {'params': self.netG.loc, 'lr': self.configs.lr_g * 10}],
                                       lr=self.configs.lr_g)

        # Setup schedulers for learning rate decay
        self.schedulerD = StepLR(optimizer=self.optimizerD,
                                 step_size=self.configs.decay_step,
                                 gamma=1)  # no decay for netD
        self.schedulerG = StepLR(optimizer=self.optimizerG,
                                 step_size=self.configs.decay_step,
                                 gamma=self.configs.decay_gamma)

        # Initialize networks

        # Initialize netD randomly
        nn.init.xavier_uniform_(self.netD.outer.weight)
        nn.init.xavier_uniform_(self.netD.linear.weight)
        self.netD.outer.bias.data.fill_(0.01)

        # Initialize netG randomly or using data
        if not self.configs.rand_init:
            loc, scale = self.SS(self.real)
            rho = self.kendall(self.real)
            init_cov = (scale @ rho @ scale).to(self.device)
            init_loc = loc.to(self.device)

            #fitted = MCD().fit(self.real.cpu())
            #MCD_cov = torch.Tensor(fitted.covariance_).to(self.device)
            #MCD_loc = torch.Tensor(fitted.location_).to(self.device)

            self.netG.precision.weight.data = torch.linalg.inv(torch.linalg.cholesky(init_cov, upper=True)).T
            self.netG.loc.data = -init_loc
        else:
            nn.init.xavier_uniform_(self.netG.precision.weight)

            # Do NOT use torch.median(dim=...), non-deterministic and is not controlled by global random seed
            # self.netG.loc.data = -self.real.median(dim=0).values
            self.netG.loc.data = -self.real.quantile(q=0.5, dim=0)

        if self.configs.no_loc:
            self.netG.loc.data = -self.mu
            
        # Initialize Tensorboard writer

        # Initialize tracker
        self.tracker = pd.DataFrame(index=range(self.configs.n_iter + 1),
                                    columns=['dloss', 'gloss', 'dpen'])
        self.tracker[:1] = 0

    # Method for sampling real data
    def sampler_real(self):
        p = self.configs.p
        n = self.configs.n
        eps = self.configs.eps
        n_Q = round(eps * n)

        mu = torch.ones(p) * self.configs.loc
        Sigma = torch.eye(p)

        if self.configs.cov == 'ar':
            for i in range(p):
                for j in range(p):
                    Sigma[i][j] = 2 ** (-abs(i - j))

        # Generate Uncontaminated data on cpu
        P_0 = MultivariateNormal(loc=mu, covariance_matrix=Sigma). \
            sample(torch.Size([n - n_Q])).to('cpu')

        # Generate contamination data on cpu
        Q = torch.empty(n_Q, p, device='cpu')\

        if self.configs.Q == 'far_cluster':
            # The second type
            Q.normal_(mean=0, std=5 ** .5)
            chi2 = torch.empty(n_Q, 1,  device='cpu').normal_() ** 2
            Q /= chi2 ** .5
            Q += 5 * torch.ones(p, device='cpu')
        elif self.configs.Q == 'far_point':
            # Single point
            Q.fill_(5)
        elif self.configs.Q == 'close_cluster':
            # The first type
            Q.normal_(mean=0, std=1/3 ** .5)
            chi2 = torch.empty(n_Q, 1,  device='cpu').normal_() ** 2
            Q /= chi2 ** .5
            Q += 2.25 * torch.tensor([1 if i % 2 == 0 else -1 for i in range(p)], device='cpu')
        else:
            Q.normal_(median=5, sigma=5 ** .5)

        # Concat and shuffle data, pass to target device
        X = torch.cat([P_0, Q], dim=0).to(self.device)
        ind_shuffle = torch.randperm(X.shape[0])
        X = X[ind_shuffle]
        return X, mu.to(self.device), Sigma.to(self.device)

    # Method for sampling noise
    def sampler_fake(self, size=None):
        if not size:
            return torch.empty(self.configs.batch_size, self.configs.p, device=self.device).normal_()
        else:
            return torch.empty(size, self.configs.p, device=self.device).normal_()
    
    # Utils
    @staticmethod
    def kendall(x):
        n, p = x.shape
        x = x.cpu().numpy()
        tao = np.zeros((p, p))
        for j in range(p):
            for k in range(j + 1):
                tao[j, k] = np.sin(np.pi / 2 * kendalltau(x[:, j], x[:, k])[0])
                tao[k, j] = tao[j, k]
        return torch.tensor(tao, dtype=torch.float32)
    
    @staticmethod
    def SS(x):
        n, p = x.shape
        x = x.cpu()
        med = x.quantile(q=0.5, dim=0)
        x0 = x - med
        S = abs(x0).quantile(q=0.5, dim=0) / norm.ppf(3/4)
        return med, S.diag()

    # Loss functions
    # (free from device choice as long as hg and hgx are on the same device)
    @staticmethod
    def rKL_loss(hx, hgx):
        hx = hx[hx >= -20]  # drop overflow points
        return 1 - (-hx).exp().mean() - hgx.mean().clip(min=-9)  # value is clipped from above at 10

    @staticmethod
    def JS_loss(hx, hgx):
        return 2 * math.log(2) + \
            F.logsigmoid(hx).mean() + \
            F.logsigmoid(-hgx).mean()

    @staticmethod
    def hinge_loss(hx, hgx):
        # E_real min(1, h(x)) + E_fake min(1, -h(x))
        return -F.relu(1. - hx).mean() - \
            F.relu(1 + hgx).mean() + 2

    def loss_select(self, name):
        dic = {
            'rKL': (self.rKL_loss, self.rKL_loss),
            'JS': (self.JS_loss, self.JS_loss),
            'hinge_cal': (self.hinge_loss, self.rKL_loss),
            'hinge': (self.hinge_loss, self.hinge_loss)
        }
        return dic.get(name, 'Not a supported loss function.')

    # Train
    def train(self):
        digit_show = str(len(str(self.configs.n_iter)))
        formatted = '0' + digit_show + 'd'

        # Training
        start_time = time.time()
        for i in range(self.configs.n_iter):
            for j, x in enumerate(self.data_loader):
                # Update the discriminator
                for i_d in range(self.configs.n_iter_d):
                    self.netD.zero_grad()

                    fake = self.sampler_fake()
                    h_real = self.netD(self.netG(x).detach())
                    h_fake = self.netD(fake)

                    # Penalty on netD
                    if not self.configs.l1:
                        pen_quad = self.netD.outer.weight.norm(2)
                        pen_lin = self.netD.linear.weight.norm(2) / math.sqrt(self.configs.p)
                        pen = pen_quad + pen_lin
                    else:
                        pen_quad = self.netD.outer.weight.norm(1)
                        pen_lin = self.netD.linear.weight.norm(1)
                        pen = pen_quad + pen_lin

                    D_loss_ = -self.loss_d(h_real, h_fake)
                    D_loss = D_loss_ + self.configs.lambda_d * pen

                    D_loss.backward()

                    # Grad clipping
                    if self.configs.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.netD.parameters(),
                                                       self.configs.grad_clip)
                    self.optimizerD.step()

                # update the generator
                for i_g in range(self.configs.n_iter_g):
                    self.netG.zero_grad()
                    
                    fake = self.sampler_fake()
                    h_real = self.netD(self.netG(x))
                    h_fake = self.netD(fake).detach()

                    gloss = self.loss_g(h_real, h_fake)

                    gloss.backward()

                    # Grad clipping
                    if self.configs.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.configs.grad_clip)
                    self.optimizerG.step()

            self.tracker.dloss[i + 1] = -D_loss_.item()  # L(D_{t-1}, G_{t-1})
            self.tracker.dpen[i + 1] = self.configs.lambda_d * pen.item()  # pen(D_{t-1})
            self.tracker.gloss[i + 1] = gloss.item()  # L(D_{t}, G_{t-1})

            # Learning rate decay
            self.schedulerG.step()
            self.schedulerD.step()

            # Gather estimation error after each epoch
            mu_hat = -self.netG.loc
            tmp = torch.linalg.inv(self.netG.precision.weight)
            Sigma_hat = tmp @ tmp.T

            mu_err = (mu_hat - self.mu).norm(2)
            if not self.configs.l1:
                Sigma_err = (Sigma_hat - self.Sigma).detach().norm(2)
            else:
                Sigma_err = (Sigma_hat - self.Sigma).detach().norm(float('inf'))

            epoch_time = time.time() - start_time

            # Display result and output
            if i % self.configs.display_gap == 0:
                epoch_time = time.time() - start_time
                formatted_time = "{:.2f}s".format(epoch_time)
                print('Epoch', format(i, formatted),
                      '|', formatted_time,
                      '| cov:', round(Sigma_err.item(), 4),
                      '| loc:', round(mu_err.item(), 4),
                      '| D_loss:', round(self.tracker.dloss[i + 1], 4),
                      '| D_pen:', round(self.tracker.dpen[i + 1], 4),
                      '| G_loss:', round(self.tracker.gloss[i + 1], 4)
                      )
                start_time = time.time()

    # Method for evaluate the estimation and saving results
    def evaluate(self, write_to_csv=False):
        h_real = self.netD(self.netG(self.real)).detach()
        h_fake = self.netD(self.sampler_fake(size=self.configs.n)).detach()
        loss = self.loss_d(h_real, h_fake)

        mu_hat = -self.netG.loc.detach()
        tmp = torch.linalg.inv(self.netG.precision.weight.detach())
        Sigma_hat = tmp @ tmp.T
        mu_err = (mu_hat - self.mu).norm(2)

        val, _ = torch.linalg.eig(Sigma_hat - self.Sigma)
        OP_err = abs(val).max()  # op norm
        F_err = sum(abs(val) ** 2) ** .5  # F norm
        max_err = (Sigma_hat - self.Sigma).norm(float('inf'))

        print('\n sup_{h} K(P, P_hat; h) = ' + str(loss.item()))
        print('\n||mu_hat - mu||_2 = ' + str(mu_err.item()))
        print('||Sigma_hat - Sigma||_F = ' + str(F_err.item()))
        print('||Sigma_hat - Sigma||_op = ' + str(OP_err.item()))
        print('||Sigma_hat - Sigma||_max = ' + str(max_err.item()))
        print('\n')

        # Append value of errors to a csv file as a new row, create file if it doesn't exist.
        if write_to_csv:
            # writing to csv file
            filename = 'metrics.csv'
            with open(filename, 'a+', newline='') as csvfile:
                # writing the data rows
                csvwriter = csv.writer(csvfile)
                new_row = [str(x) for x in [mu_err.item(), F_err.item(), OP_err.item(), max_err.item()]]
                csvwriter.writerow(new_row)
