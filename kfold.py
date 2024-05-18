import argparse
import random
import numpy as np
from torch import backends,cuda,manual_seed

from cross_validation.k_fold import KFoldTrainer

def main(config_path,start_fold):
    # make sure the result is reproducible
    backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
    # Create an instance of KFoldTrainer
    trainer = KFoldTrainer(
        config_path=config_path,
    )

    # Start k-fold cross-validation
    trainer.k_fold(start_fold=start_fold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KFoldTrainer with specified config path.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--start_fold', type=int, default=0, help='The fold number to start training from.')
    args = parser.parse_args()
    main(args.config_path,args.start_fold)
