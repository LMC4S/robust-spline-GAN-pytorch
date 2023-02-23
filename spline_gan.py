import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import StepLR

import pandas as pd
import math

from tensorboardX import SummaryWriter


# Loss
def rKL_loss(hx, hgx):
    return 1 - (-hx).exp().mean() - hgx.mean()


def JS_loss(hx, hgx):
    return 2 * math.log(2) - \
           torch.log(1 + torch.exp(-hx)).mean() - \
           torch.log(1 + torch.exp(hgx)).mean()


def H2_loss(hx, hgx):
    return -(-hx / 2).exp().mean() - (hgx / 2).exp().mean() + 2


def hinge_loss(hx, hgx):
    # E_real min(1, h(x)) + E_fake min(1, -h(x))
    return -F.relu(1. - hx).mean() - \
           F.relu(1 + hgx).mean() + 2


def loss_select(name):
    dic = {
        'rKL': (rKL_loss, rKL_loss),
        'JS': (JS_loss, JS_loss),
        'hinge_cal': (hinge_loss, rKL_loss),
        'hinge': (hinge_loss, hinge_loss),
        'H2': (H2_loss, H2_loss)
    }
    return dic.get(name, 'Not a supported loss function.')


# Models
class Generator(nn.Module):
    # General covariance sturcture
    def __init__(self, p):
        super(Generator, self).__init__()
        self.p = p
        # weight = Cov^{hat}^{1/2}, bias = mu_hat
        self.params = nn.Linear(p, p, bias=True)

    def forward(self, x):
        # g(x) = Cov^{1/2}x + mu
        return self.params(x)


class Discriminator(nn.Module):
    # Quadratic full-interaction spline
    def __init__(self, p, knots):
        super(Discriminator, self).__init__()
        self.p = p
        self.knots = knots  # k-dim list of floats or tensor floats
        self.aug_linear_dim = (1+len(self.knots)) * p
        self.acf = nn.ReLU()
        self.outer = nn.Bilinear(self.aug_linear_dim,
                                 self.aug_linear_dim,
                                 1, bias=True)

    def forward(self, x):
        aug = self.acf(torch.cat([x-knot for knot in self.knots], dim=1))
        aug = torch.cat([x, aug], dim=1)
        return self.outer(aug, aug)


# Robust spline GAN
class SplineGAN:
    def __init__(self, config):
        self.config = config

        # Set universal random seed for reproducibility
        torch.manual_seed(self.config.seed)

        # Generate real data X under config model setting
        self.real, self.mu, self.Sigma = self.sampler_real()

        # Setup networks and loss
        self.netG = Generator(self.config.p)
        self.netD = Discriminator(self.config.p, knots=[-1, 0, 1])
        print("Using", torch.cuda.device_count(), "GPUs for training.")
        self.netG = nn.DataParallel(self.netG)
        self.netD = nn.DataParallel(self.netD)
        self.netG.to('cuda')
        self.netD.to('cuda')

        self.loss_d, self.loss_g = loss_select(self.config.loss)

        # Setup optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.lr_d)
        if self.config.no_loc:
            self.optimizerG = optim.Adam([{'params': self.netG.params.weight}],
                                         lr=self.config.lr_g)
        else:
            self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.config.lr_g)

        # Setup schedulers for learning rate decay
        self.schedulerD = StepLR(optimizer=self.optimizerD,
                                 step_size=self.config.decay_step,
                                 gamma=self.config.decay_gamma)
        self.schedulerG = StepLR(optimizer=self.optimizerG,
                                 step_size=self.config.decay_step,
                                 gamma=self.config.decay_gamma)

        # Initialize networks
        nn.init.xavier_uniform_(self.netD.module.outer.weight)
        self.netD.module.outer.bias.data.fill_(0.01)

        if self.config.use_sample_cov:
            sample_cov = self.real.T.cov()
            sample_loc = self.real.mean(dim=0)
            self.netG.module.params.weight.data = torch.linalg.cholesky(sample_cov)
            self.netG.module.params.bias.data = sample_loc
        else:
            nn.init.xavier_uniform_(self.netG.module.params.weight)
            if self.config.no_loc:
                self.netG.module.params.bias.data = self.mu
            else:
                self.netG.module.params.bias.data = self.real.median(dim=0).values

        # Initialize Tensorboard writer
        if self.config.out_dir is not None:
            self.writer = SummaryWriter(self.config.out_dir)
        else:
            self.writer = SummaryWriter()
        self.writer_head = 0  # writer's current head index

    # Sample real data
    def sampler_real(self):
        p = self.config.p
        n = self.config.n
        eps = self.config.eps
        n_Q = round(eps * n)
        mu = torch.ones(p)*self.config.loc
        Sigma = torch.eye(p)
        if self.config.cov == 'ar':
            for i in range(p):
                for j in range(p):
                    Sigma[i][j] = 2**(-abs(i-j))

        P_0 = MultivariateNormal(loc=mu, covariance_matrix=Sigma). \
            sample(torch.Size([n])).cuda()

        if self.config.Q == 'far_cluster':
            Q = torch.cuda.FloatTensor(n_Q, p).normal_(mean=5, std=5**.5)
        elif self.config.Q == 'far_point':
            Q = torch.cuda.FloatTensor(n_Q, p).normal_(mean=5, std=1e-5)
        elif self.config.Q == 'close_cluster':
            Q = torch.cuda.FloatTensor(n_Q, p).normal_(mean=1.5, std=5**.5)
        elif self.config.Q == 'close_point':
            Q = torch.cuda.FloatTensor(n_Q, p).normal_(mean=1.5, std=1e-5)
        else:
            Q = torch.cuda.FloatTensor(n_Q, p).normal_(mean=0, std=5**.5)

        return torch.cat([P_0, Q], dim=0), mu.cuda(), Sigma.cuda()

    # Sample noise for generating fake data
    def sampler_fake(self):
        return torch.cuda.FloatTensor(self.config.fake_size, self.config.p).normal_()

    # Train
    def train(self):
        _n_iter_d = self.config.n_iter_d

        digit_show = str(len(str(self.config.n_iter)))
        formatted = '0' + digit_show + 'd'

        tracker = pd.DataFrame(index=range(self.config.n_iter + 1),
                               columns=['dloss', 'gloss', 'dpen'])
        tracker[:1] = 0

        # Training
        for i in range(self.config.n_iter):
            # Warm-up setting for the discriminator
            if i == 0 and not self.config.no_warm_up:
                print('\nDiscriminator warm-up training...\n')
                n_iter_d = int(1 / self.config.lr_d)
            else:
                n_iter_d = _n_iter_d

            # Update the discriminator
            for i_d in range(n_iter_d):
                self.netD.zero_grad()

                fake = self.netG(self.sampler_fake())
                h_real = self.netD(self.real)
                h_fake = self.netD(fake.detach())
                pen = self.netD.module.outer.weight.norm(2)
                D_loss_ = -self.loss_d(h_real, h_fake)
                D_loss = D_loss_ + self.config.lambda_d * pen

                D_loss.backward()
                if not self.config.no_norm_clip:
                    torch.nn.utils.clip_grad_norm_(self.netD.parameters(), .5)
                self.optimizerD.step()

            self.writer.add_scalar('Grad/outer_weight',
                                   self.netD.module.outer.weight.grad.norm(2).item(),
                                   i + self.writer_head)
            self.writer.add_scalar('Grad/outer_bias',
                                   self.netD.module.outer.bias.grad.norm(2).item(),
                                   i + self.writer_head)
            self.writer.add_scalar('Pen/netD', pen.item(), i + self.writer_head)
            self.writer.add_scalar('Loss/netD', -D_loss_.item(), i + self.writer_head)

            # update the generator
            for i_g in range(self.config.n_iter_g):
                self.netG.zero_grad()

                fake = self.netG(self.sampler_fake())
                h_real = self.netD(self.real).detach()
                h_fake = self.netD(fake)
                gloss = self.loss_g(h_real, h_fake)

                gloss.backward()
                if not self.config.no_norm_clip:
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), .5)
                self.optimizerG.step()

            self.writer.add_scalar('Loss/netG', gloss.item(), i + self.writer_head)
            self.writer.add_scalar('Grad/sigma_hat',
                                   self.netG.module.params.weight.grad.norm(2).item(),
                                   i + self.writer_head)
            self.writer.add_scalar('Grad/mu_hat', self.netG.module.params.bias.grad.norm(2).item(),
                                   i + self.writer_head)
            self.writer.add_scalar('h(x)/real', h_real.median().item(), i + self.writer_head)
            self.writer.add_scalar('h(x)/fake', h_fake.median().item(), i + self.writer_head)

            tracker.dloss[i + 1] = -D_loss_.item()  # L(D_{t-1}, G_{t-1})
            tracker.dpen[i + 1] = self.config.lambda_d * pen.item()  # pen(D_{t-1})
            tracker.gloss[i + 1] = gloss.item()  # L(D_{t}, G_{t-1})

            # Learning rate decay
            self.schedulerG.step()
            self.schedulerD.step()

            # Estimation error after one iteration
            mu_hat = self.netG.module.params.bias
            Sigma_hat = self.netG.module.params.weight @ self.netG.module.params.weight.T
            mu_err = (mu_hat - self.mu).norm(2)
            Sigma_err = (Sigma_hat - self.Sigma).detach().norm(2)
            self.writer.add_scalar('Error/mu_l2',
                                   mu_err.item(), i + self.writer_head)
            self.writer.add_scalar('Error/Sigma_fro',
                                   Sigma_err.item(), i + self.writer_head)

            # Display result and output
            if i % self.config.display_gap == 0:
                print('Epoch', format(i, formatted),
                      '| Sigma_err:', round(Sigma_err.item(), 5),
                      '| mu_err:', round(mu_err.item() ** 2, 5),
                      '| D_loss:', round(tracker.dloss[i + 1], 5),
                      '| D_pen:', round(tracker.dpen[i + 1], 5),
                      '| G_loss:', round(tracker.gloss[i + 1], 5)
                      )

        self.writer_head += i  # set writer's head to current index

    def evaluate(self, write_to_csv=False):
        mu_hat = self.netG.module.params.bias
        Sigma_hat = self.netG.module.params.weight @ self.netG.module.params.weight.T
        mu_err = (mu_hat - self.mu).norm(2)

        val, _ = torch.linalg.eig(Sigma_hat - self.Sigma)
        OP_err = abs(val).max()  # op norm
        F_err = sum(abs(val)**2)**.5  # F norm

        print('\n||mu_hat - mu||_2: ' + str(mu_err.item()))
        print('||Sigma_hat - Sigma||_F: ' + str(F_err.item()))
        print('||Sigma_hat - Sigma||_op: ' + str(OP_err.item()))
        print('\n')

        # Append value of errors to a csv file as a new row, create file if it doesn't exist.
        if write_to_csv and self.config.out_dir is not None:
            isExist = os.path.exists(self.config.out_dir)
            if not isExist:
                os.makedirs(self.config.out_dir)
            # writing to csv file
            filename = self.config.out_dir + 'err.csv'
            with open(filename, 'a+', newline='') as csvfile:
                # writing the data rows
                csvwriter = csv.writer(csvfile)
                new_row = [str(x) for x in [mu_err.item(), F_err.item(), OP_err.item()]]
                csvwriter.writerow(new_row)

    def save_to_pt(self, pt_file=None):
        if pt_file is None:
            pt_file = self.config.out_dir+'model.pt'
        print(f'\nSaving configs and model state dicts to {pt_file}... (could overwrite)')
        # Process configs, models, optimizers, etc. for resuming training and/or inference
        dic = self.__dict__.copy()

        model_vars = ['netG', 'netD', 'schedulerD', 'schedulerG',
                      'optimizerG', 'optimizerD']
        for var in model_vars:
            # save models' state dict and remove models (some are non-serialized)
            # see https://pytorch.org/tutorials/beginner/saving_loading_models.html
            dic[var+'_state_dict'] = dic[var].state_dict()
            del dic[var]

        # Remove queue object, which is non-serialized.
        del dic['writer']

        # Save to pickle file
        torch.save(dic, self.config.out_dir+'model.pt')
        print('\nSaved.')

    def load_from_pt(self, pt_file):
        print(f'\nLoading from {pt_file}, overwriting existing attributes.')
        checkpoint = torch.load(pt_file)
        model_vars = ['netG', 'netD', 'schedulerD', 'schedulerG',
                      'optimizerG', 'optimizerD']
        for var in model_vars:
            getattr(self, var).load_state_dict(checkpoint[var+'_state_dict'])

        for key in checkpoint:
            setattr(self, key, checkpoint[key])

        print('\nLoaded.')
