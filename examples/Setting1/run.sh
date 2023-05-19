python train.py --p 10 --n 2000 --eps 0.1  \
--Q close_cluster \
--batch_size 2000 --n_iter 100 \
--decay_step 1 --decay_gamma 0.95 \
--n_iter_d 10 --n_iter_g 1 \
--lr_g 0.05 --lr_d 0.01 \
--cpu --seed 0 --loss JS > setting1.log