import torch
from spline_gan import SplineGAN
from config import Config

# Create a null SplineGAN instance with default config if not run as a script.  ```from train import model```
# config is parsed from command line input if run as a script.
configs = Config().parse()
if not configs.cpu and torch.cuda.is_available():
    print('Training with CUDA gpu:' + str(configs.cuda_id))
    configs.device = 'cuda:' + str(configs.cuda_id)
else:
    print('Training with cpu.')
    configs.device = 'cpu'


print(configs.msg)
model = SplineGAN(configs)

if __name__ == '__main__':
    # Train the model, and evaluate errors.
    # Write errors to csv file if '--out_dir' is provided.
    model.train()
    model.evaluate(write_to_csv=True)

    try:
        import session_info
        print(session_info.show())
    except ModuleNotFoundError:
        print('\nSession info is not available.\n')
