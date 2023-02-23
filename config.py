import argparse
import math
import time


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='configurations.')

        self.parser.add_argument('--n_iter', type=int, default=8000, help='int: total epochs')
        self.parser.add_argument('--decay_step', type=int, default=50, help='int: lr decay gap steps')
        self.parser.add_argument('--p', type=int, default=50, help='int: dimension')
        self.parser.add_argument('--n', type=int, default=50000, help='int: sample size')
        self.parser.add_argument('--fake_size', type=int, default=50000, help='int: generator noise sample size')
        self.parser.add_argument('--n_iter_g', type=int, default=1, help='int: G iteration steps within one epoch')
        self.parser.add_argument('--n_iter_d', type=int, default=5, help='int: D iteration steps within one epoch')
        self.parser.add_argument('--display_num', type=int, default=100, help='int: number of output prints')
        self.parser.add_argument('--seed', type=int, default=None, help='Random seed')

        self.parser.add_argument('--loc', type=float, default=0, help='float: location for all p dimension')
        self.parser.add_argument('--decay_gamma', type=float, default=0.95, help='float[0,1]: lr decay rate')
        self.parser.add_argument('--lambda_d', type=float, default=0.025, help='float: pen lvl for D')
        self.parser.add_argument('--lr_d', type=float, default=5e-4, help='float: lr D')
        self.parser.add_argument('--lr_g', type=float, default=5e-3, help='float: lr G')
        self.parser.add_argument('--eps', type=float, default=0.0, help='float[0, 1]: contamination proportion')

        self.parser.add_argument('--loss', type=str, default='hinge', help='str: Loss',
                                 choices=['hinge', 'hinge_cal', 'JS', 'rKL'])
        self.parser.add_argument('--Q', type=str, default='far_cluster', help='str: type of contamination Q',
                                 choices=['far_cluster', 'far_point', 'close_cluster', 'close_point'])
        self.parser.add_argument('--cov', type=str, default='ar', choices=['ar', 'id'],
                                 help='str: model covariance matrix')
        self.parser.add_argument('--out_dir', type=str, default=None, help='str: output path')

        self.parser.add_argument('--no_loc', default=False, action='store_true',
                                 help='feature: fix location mu at true value')
        self.parser.add_argument('--use_sample_cov', default=False, action='store_true',
                                 help='feature: use sample cov for generator initialization')
        self.parser.add_argument('--no_warm_up', default=False, action='store_true',
                                 help='feature: no warm-up train for discriminator')
        self.parser.add_argument('--no_norm_clip', default=False, action='store_true',
                                 help='feature: no gradient clipping')

    def parse(self, command=None):
        if command is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(command)
        message = ''
        message += '----------------- Configs ---------------\n'
        message += ' '.join(f'{k}={v}' for k, v in vars(args).items())
        message += '\n\n[Non-default Values]: '
        message += ' '.join(f'{k}={v}' for k, v in vars(args).items() if v != self.parser.get_default(k))
        message += '\n----------------- Configs End -------------------'
        args.msg = message

        args.display_gap = max(int(args.n_iter/args.display_num), 1)

        # Compute lambda_d
        rate = math.sqrt(args.p / args.n)
        args.lambda_d *= rate

        # Make random seed if not provided
        if args.seed is None:
            args.seed = time.time()

        # Make sure the provided output dir is a folder
        if args.out_dir is not None and args.out_dir[-1] != '/':
            args.out_dir += '/'

        return args
