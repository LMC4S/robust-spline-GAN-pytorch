
# PyTorch Implementation of logit f-GAN and Hinge GAN for Robust Location and Covariance Estimation

[Tractable and Near-Optimal Adversarial Algorithms for Robust Estimation in Contaminated Gaussian Models](https://arxiv.org/abs/2112.12919)

For simultaneous robust estimation of location and covariance matrix under Huber's contamination model where data is contaminated by a proportion of arbitrary outliers. 

The discriminator employs truncated linear spline with pairwise interactions, resulting in tractable training (concave surface) and desirable theoretical guarantees.

For large models (p > 50), training on a cuda gpu is preferred.

------
## Run on cpu

```
# JS logit f-GAN training on cpu

python train.py --p 10 --n 2000 --eps 0.1  \
--Q close_cluster \
--batch_size 2000 --n_iter 100 \
--decay_step 1 --decay_gamma 0.95 \
--n_iter_d 10 --n_iter_g 1 \
--lr_g 0.05 --lr_d 0.01 \
--cpu --seed 0 --loss JS \


# Try rKL logit f-GAN or hinge GAN \ 
  with loss "rKL" or "hinge"
```

Above code replicates the R implementation in https://github.com/LMC4S/robust-spline-GAN for JS and rKL logit f-GAN as well as the hinge GAN. 

## Run on CUDA gpu
```
# JS logit f-GAN 
# Using default (adaptive) hyperparameters

python train.py --p 100 --n 20000 --eps 0.2 \
--Q close_cluster \
--rand_init \
--cuda_id 0 --seed 0 --loss JS \
--adaptive
```
Above code works for training with p=100 and n=20000 on a single gpu. For other models considered in the paper please make sure that "--adaptive" and "--rand_init" options are used. See *config.py* for details.
