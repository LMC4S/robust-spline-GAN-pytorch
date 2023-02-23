import torch
from spline_gan import SplineGAN
from config import Config

# Create a null SplineGAN instance with default config if not run as a script.  ```from train import model```
# config is parsed from command line input if run as a script.
assert torch.cuda.is_available(), 'A CUDA GPU is required.'
config = Config().parse()
model = SplineGAN(config)

if __name__ == '__main__':
    print(config.msg)

    # Train the model, write tensorboard event files, evaluate errors.
    # Write errors to csv file if '--out_dir' is provided.
    model.train()
    model.evaluate(write_to_csv=True)

    try:
        import session_info
        print(session_info.show())
    except ModuleNotFoundError:
        print('\nSession info is not available.\n')
