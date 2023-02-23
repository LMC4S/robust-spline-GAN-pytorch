Training on Single GPU
```{bash}
CUDA_VISIBLE_DEVICES=0 python3 train.py --p 40 --n 50000 --eps 0.2 --out_dir output --seed 0 > out.log 
```
Use tensorboard for visiualization of the training process.
```
tensorboard --logdir=<path-to-tfevent-file>
```