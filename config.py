import argparse
import math
import time


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='configurations.')

        self.parser.add_argument('--n_iter', type=int, default=150, help='int: number of epochs')
        self.parser.add_argument('--decay_step', type=int, default=10, help='int: lr decay steps')
        self.parser.add_argument('--p', type=int, default=100, help='int: dimension')
        self.parser.add_argument('--n', type=int, default=20000, help='int: sample size')
        self.parser.add_argument('--batch_size', type=int, default=1000,
                                 help='int: mini-batch size')
        self.parser.add_argument('--n_iter_g', type=int, default=4, help='int: G iteration steps')
        self.parser.add_argument('--n_iter_d', type=int, default=20, help='int: D iteration steps')
        self.parser.add_argument('--display_num', type=int, default=50, help='int: control report frequency in log')
        self.parser.add_argument('--seed', type=int, default=None, help='int: random seed')

        self.parser.add_argument('--loc', type=float, default=0, help='float: true location for all dimensions')
        self.parser.add_argument('--decay_gamma', type=float, default=0.5, help='float[0., 1.]: lr decay rate')
        self.parser.add_argument('--lambda_d', type=float, default=-1, help='float: penalty level for D')
        self.parser.add_argument('--lr_d', type=float, default=2e-4, help='float: lr D')
        self.parser.add_argument('--lr_g', type=float, default=1e-3, help='float: lr G')
        self.parser.add_argument('--eps', type=float, default=0.2, help='float[0., .5): contamination proportion')
        self.parser.add_argument('--grad_clip', default=1,
                                 help='float[0., +inf]: gradient clipping level')

        self.parser.add_argument('--loss', type=str, default='hinge', help='str: Loss',
                                 choices=['hinge', 'hinge_cal', 'JS', 'rKL'])
        self.parser.add_argument('--Q', type=str, default='close_cluster', help='str: type of contamination Q',
                                 choices=['far_cluster', 'far_point', 'close_cluster'])
        self.parser.add_argument('--cov', type=str, default='ar', choices=['ar', 'id'],
                                 help='str: true covariance matrix')
        self.parser.add_argument('--out_dir', type=str, default=None, help='str: path for output logs')
        self.parser.add_argument('--pen_type', type=str, default="l2", choices=['l1', 'l2'],
                                 help='str: type of penalty')

        self.parser.add_argument('--no_loc', default=False, action='store_true',
                                 help='feature: fix location at true value, estimate cov only.')
        self.parser.add_argument('--cpu', default=False, action='store_true', help='feature: Train with cpu')
        self.parser.add_argument('--rand_init', default=False, action='store_true',
                                 help='feature: initialize generator by Xavier Uniform instead of using data ')
        self.parser.add_argument('--adaptive', default=False, action='store_true',
                                 help='feature: Use lr adaptive to model settings')

        self.parser.add_argument('--cuda_id', default=0, type=int,
                                 help='int: The id of gpu to train on (when multiple gpus are available)')
        
    def parse(self, command=None):
        if command is None:
            args = self.parser.parse_known_args()[0]
        else:
            args = self.parser.parse_known_args(command)[0]

        # Store penalty type
        args.l1 = (args.pen_type == "l1")

        # Set default lambda_d by loss if not provided
        lambdas = {'rKL': 0.3, 'hinge': 0.1, 'JS': 0.025}
        if args.lambda_d == -1:
            args.lambda_d = lambdas[args.loss]

        if args.adaptive:
            # For replication of our experiment results in paper please use adaptive lr

            # Adaptive to different sample size n from default 20000
            args.n_iter = int(args.n_iter * 50000 / args.n)
            args.decay_step = int(args.decay_step * 50000 / args.n)

            # Larger lr is appreciated in settings with smaller p and n
            if args.p <= 50:
                args.lr_g *= 5
                args.lr_d *= 5
                if args.p <= 10:
                    args.lr_g *= 2
                    args.lr_d *= 2
            if args.n <= 5000:
                args.lr_g *= 2
                args.lr_d *= 2
            
        # Make random seed if not provided
        if args.seed is None:
            args.seed = time.time()
        
        message = ''
        message += '------------ Configs (after adaptation) --------------------\n'
        message += ' '.join(f'--{k} {v}' for k, v in vars(args).items())
        message += '\n\n[Non-default Values]: '
        message += ' '.join(f'{k}={v}' for k, v in vars(args).items() if v != self.parser.get_default(k))
        message += '\n------------ Configs End -----------------------------------'
        args.msg = message

        args.display_gap = max(int(args.n_iter/args.display_num), 1)

        # Compute lambda_d
        if not args.l1:
            rate = math.sqrt(args.p ** 2 / args.n)
        else:
            rate = math.sqrt(math.log(args.p) / args.n)

        args.lambda_d *= rate

        return args
