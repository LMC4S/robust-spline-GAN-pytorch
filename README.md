# Pytorch implementation for robust logit fGAN and hinge GAN
Pytorch implementation for [Tractable and Near-Optimal Adversarial Algorithms for Robust Estimation in Contaminated Gaussian Models](https://arxiv.org/abs/2112.12919) and https://github.com/LMC4S/robust-spline-GAN.


------
## Environment
```
# python modules
Python 3.9.16

numpy==1.23.5
pandas==1.5.2
scipy==1.9.3
tensorboardX==2.6
torch==1.13.1

# CUDA 
CUDA 11.7

# System
Ubuntu 20.04.4 LTS (GNU/Linux 5.14.0-1056-oem x86_64)
```

## Use case
```
python3 train.py --p 40 --n 50000 --eps 0.2 --out_dir output 
```
