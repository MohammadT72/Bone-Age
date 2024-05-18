# Third-party library imports
import os
import pandas as pd
import matplotlib.pyplot as plt

# Monai
from monai.data import DataLoader, Dataset, CacheDataset, PersistentDataset
# Wandb
import wandb

class DataLoaderWrapper:
    """
    A wrapper class for creating data loaders and handling dataset selection.

    Methods:
    --------
    make_dataloader(batch_size, dataset_name='dataset', cache_rate=1.0, num_workers=0):
        Creates data loaders for training and validation datasets.
    select_data_list(fold, frac):
        Selects the training, validation, and test data for a specific fold.
    check_data(fold):
        Checks and prints the preprocessing steps and data statistics for the given fold.
    plot_data(samples, fold, name='train'):
        Plots a subset of the data samples and logs them to Weights & Biases.
    """
    def __init__(self,):
        self.train_data_list=[]
        self.val_data_list=[]
        self.test_data_list=[]
    def get_test_data(self,):
      # Select the test data
      if self.config==None:
        self.read_config()
      self.prepare_settings()
      # Drop columns specifing fold splits
      cols_to_drop = self.data.columns[self.data.columns.str.contains('fold')]
      self.data = self.data.drop(columns=cols_to_drop)
      test_data_list = self.data[self.data['test'] == True].to_dict('records')
      self.test_data_list = test_data_list[:int(len(test_data_list))]
      test_ds = Dataset(self.test_data_list, self.val_transforms)
      self.test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
      return self.test_loader
    def make_dataloader(self, batch_size, dataset_name='dataset', cache_rate=1.0, num_workers=0):
        """
        Creates data loaders for training and validation datasets.

        Parameters:
        ----------
        batch_size : int
            The batch size for the data loaders.
        dataset_name : str, optional
            The name of the dataset type ('dataset' or 'cache_dataset') (default is 'dataset').
        cache_rate : float, optional
            The caching rate for CacheDataset (default is 1.0).
        num_workers : int, optional
            The number of worker threads to use for data loading (default is 0).
        """
        # Create datasets based on the specified dataset name
        if dataset_name == 'dataset':
            train_ds = Dataset(self.train_data_list, self.train_transforms)
            val_ds = Dataset(self.val_data_list, self.val_transforms)
        elif dataset_name == 'cache_dataset':
            train_ds = CacheDataset(self.train_data_list, self.train_transforms, cache_rate=cache_rate)
            val_ds = CacheDataset(self.val_data_list, self.val_transforms, cache_rate=cache_rate)
        elif dataset_name == 'persistent_dataset':
            cache_dir=os.path.join(os.getcwd(),'cache')
            os.makedirs(cache_dir,exist_ok=True)
            train_ds = PersistentDataset(self.train_data_list, self.train_transforms,cache_dir=cache_dir)
            val_ds = PersistentDataset(self.val_data_list, self.val_transforms,cache_dir=cache_dir)
        else:
            raise ValueError(f'Dataset {dataset_name} is not in the implemented list')

        # Create data loaders for training and validation datasets
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def select_data_list(self, fold, frac):
        """
        Selects the training, validation, and test data for a specific fold.

        Parameters:
        ----------
        fold : int
            The fold number for cross-validation.
        frac : float
            The fraction of data to use for training, validation, and testing.
        """
        # Select the training data for the given fold
        train_data_list = self.data[self.data[f'fold_{fold+1}'] == 'train'].to_dict('records')
        self.train_data_list = train_data_list[:int(len(train_data_list) * frac)]

        # Select the validation data for the given fold
        val_data_list = self.data[self.data[f'fold_{fold+1}'] == 'val'].to_dict('records')
        self.val_data_list = val_data_list[:int(len(val_data_list) * frac)]

        # Select the test data
        test_data_list = self.data[self.data['test'] == True].to_dict('records')
        self.test_data_list = test_data_list[:int(len(test_data_list) * frac)]

        # Print the number of records in each dataset
        print(f'Number of records in the train data list: {len(self.train_data_list)}')
        print(f'Number of records in the validation data list: {len(self.val_data_list)}')
        print(f'Number of records in the test data list: {len(self.test_data_list)}')
    def select_data_list_final(self, frac):
        """
        Selects the training,and test data for the final training.

        Parameters:
        ----------
        frac : float
            The fraction of data to use for training, validation, and testing.
        """
        # Drop columns specifing fold splits
        cols_to_drop = self.data.columns[self.data.columns.str.contains('fold')]
        self.data = self.data.drop(columns=cols_to_drop)
        # Select the training data for the given fold
        train_data_list = self.data[self.data['test'] == False].to_dict('records')
        self.train_data_list = train_data_list[:int(len(train_data_list) * frac)]
        # Select the test data
        val_data_list = self.data[self.data['test'] == True].to_dict('records')
        self.val_data_list = val_data_list[:int(len(val_data_list) * frac)]

        # Print the number of records in each dataset
        print(f'Number of records in the train data list: {len(self.train_data_list)}')
        print(f'Number of records in the test data list: {len(self.val_data_list)}')

    def check_data(self, fold):
        """
        Checks and prints the preprocessing steps and data statistics for the given fold.

        Parameters:
        ----------
        fold : int
            The fold number for cross-validation.
        """
        if self.check:
            print('Checking the input ...')

            # Check training data
            print('--- train data')
            print('------ preprocessing')
            print("Train Transforms:")
            print(self.preprocessing_steps_train)
            sample = next(iter(self.train_loader))
            input_size = sample[self.image_key].shape
            minv = sample[self.image_key].min()
            maxv = sample[self.image_key].max()
            labels = sample[self.target_col_name]
            self.plot_data(sample, fold, name='train')
            print(f'input size: {input_size}')
            print(f'input minv: {minv}, input maxv: {maxv}')
            print(f'Sample of labels: {labels[:5]}')

            # Check validation data
            print('--- validation data')
            print('------ preprocessing')
            print("Validation Transforms:")
            print(self.preprocessing_steps_val)
            sample = next(iter(self.val_loader))
            self.plot_data(sample, fold, name='val')
            input_size = sample[self.image_key].shape
            minv = sample[self.image_key].min()
            maxv = sample[self.image_key].max()
            labels = sample[self.target_col_name]
            print(f'input size: {input_size}')
            print(f'input minv: {minv}, input maxv: {maxv}')
            print(f'Sample of labels: {labels[:5]}')

    def plot_data(self, samples, fold, name='train'):
        """
        Plots a subset of the data samples and logs them to Weights & Biases.

        Parameters:
        ----------
        samples : dict
            A dictionary of sample data.
        fold : int
            The fold number for cross-validation.
        name : str, optional
            The name of the data subset ('train' or 'val') (default is 'train').
        """
        if self.batch_size > 10:
            fig, axes = plt.subplots(2, 5, figsize=(10, 5))
            for idx in range(10):
                row = 0 if idx < 5 else 1
                col = idx if idx < 5 else idx - 5
                result = samples['path'].detach().cpu()[idx, 0, :, :]
                axes[row, col].imshow(result, cmap='gray')
                mean = result.mean()
                std = result.std()
                max = result.max()
                min = result.min()
                axes[row, col].set_title(f'Mean:{mean:.2f},Std:{std:.2f}\nMin:{min:.2f},Max:{max:.2f}')
            plt.tight_layout()
            wandb.log({f"samples_{name}_fold_{fold}": wandb.Image(fig)})