import numpy as np
from torch import device,cuda
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR

from train.helpers import Helpers
from train.validation import Validation
from train.training import Train

class TrainerWrapper(Train,Validation,Helpers):
    """
    A wrapper class for training a PyTorch model with advanced features such as early stopping,
    mixed precision training, and logging with Weights & Biases (wandb).

    Attributes:
    -----------
    best_r2s : float
        The best R^2 score achieved during training.
    best_r2s_epoch : int
        The epoch at which the best R^2 score was achieved.
    val_loss_min : float
        The minimum validation loss achieved during training.
    best_score : float
        The best score achieved for early stopping.
    counter : int
        The counter for early stopping.
    early_stop : bool
        Flag to indicate whether early stopping has been triggered.
    best_epoch : int
        The epoch at which the best model was saved.
    epoch : int
        The current epoch number.
    train_step : int
        The current training step number within an epoch.
    val_step : int
        The current validation step number within an epoch.

    Methods:
    --------
    initialize_training_settings(...)
        Initializes the training settings, including data loaders, model, optimizer, loss function, etc.
    reset_trainer()
        Resets the training state variables.
    early_stop_check(input_value, loss, model, epoch)
        Checks if training should be stopped early based on validation performance.
    train()
        Executes the training loop for one epoch.
    validation()
        Executes the validation loop.
    metric_calculation(epoch, y_pred, y)
        Calculates and logs metrics based on predictions and actual values.
    save_model(name='best')
        Saves the model checkpoint.
    start()
        Starts the training process.
    """

    def __init__(self):
        """
        Initializes the TrainerWrapper class with default values for training state variables.
        """
        # Initialize training state variables
        self.best_r2s = -1  # Best R^2 score achieved
        self.best_r2s_epoch = -1  # Epoch at which the best R^2 score was achieved
        self.val_loss_min = np.Inf  # Minimum validation loss achieved
        self.best_score = None  # Best score for early stopping
        self.counter = 0  # Counter for early stopping
        self.early_stop = False  # Flag for early stopping
        self.best_epoch = 0  # Epoch at which the best model was saved
        self.epoch = 0  # Current epoch number
        self.train_step = 0  # Current training step number
        self.val_step = 0  # Current validation step number
    def initialize_training_settings(self, train_data_loader, val_data_loader,
                                     model, optimizer, loss_function, max_epochs,
                                     target_col_name, image_key, batch_size,
                                     metric_calculator, exp_name='train',
                                     model_name='resnet18', loss_function_name='mse',
                                     initial_lr=0.001, final_lr=0.00001,
                                     scheduler_name=None, amp=True, val_interval=1,
                                     device=device('cpu'), show_interval=10,
                                     early_stop_callback=True, patience=2,
                                     early_stop_verbose=True, early_stop_delta=0,
                                     fold_num=0):
      """
        Initializes the training settings, including data loaders, model, optimizer,
          loss function, and various parameters.

        Parameters:
        ----------
        train_data_loader : DataLoader
            DataLoader for the training data.
        val_data_loader : DataLoader
            DataLoader for the validation data.
        model : torch.nn.Module
            The PyTorch model to be trained.
        optimizer : torch.optim.Optimizer
            The optimizer for training the model.
        loss_function : callable
            The loss function used for training.
        max_epochs : int
            The maximum number of epochs for training.
        target_col_name : str
            The column name for the target variable.
        image_key : str
            The key for the image data in the DataLoader.
        batch_size : int
            The batch size for training.
        metric_calculator : callable
            The metric calculator for evaluation.
        exp_name : str, optional
            The experiment name (default is 'train').
        model_name : str, optional
            The model name (default is 'resnet18').
        loss_function_name : str, optional
            The name of the loss function (default is 'mse').
        initial_lr : float, optional
            The initial learning rate (default is 0.001).
        final_lr : float, optional
            The final learning rate (default is 0.00001).
        scheduler_name : str, optional
            The name of the learning rate scheduler (default is None).
        amp : bool, optional
            Whether to use automatic mixed precision (default is True).
        val_interval : int, optional
            The interval for validation (default is 1).
        device : torch.device, optional
            The device to use for training (default is CPU).
        show_interval : int, optional
            The interval for showing training progress (default is 10).
        early_stop_callback : bool, optional
            Whether to use early stopping (default is True).
        patience : int, optional
            The patience for early stopping (default is 2).
        early_stop_verbose : bool, optional
            Whether to show verbose early stopping messages (default is True).
        early_stop_delta : float, optional
            The delta for early stopping (default is 0).
        fold_num : int, optional
            The fold number for cross-validation (default is 0).
      """
      # Store training settings
      self.train_loader = train_data_loader
      self.val_loader = val_data_loader
      self.model = model
      self.optimizer = optimizer
      self.scheduler_name=scheduler_name
      self.loss_function = loss_function
      self.max_epochs = max_epochs
      self.target_col_name=target_col_name
      self.image_key=image_key
      self.metric_calculator = metric_calculator
      self.train_num_batch = len(train_data_loader)
      self.early_stop_callback=early_stop_callback
      self.val_num_batch = len(val_data_loader)
      self.val_interval = val_interval
      self.show_interval=show_interval
      self.device=device
      self.root_dir = '/content'
      self.enabled = True if 'cuda' in device.type else False
      if not amp:
          self.enabled = False
      self.scaler = cuda.amp.GradScaler(enabled=self.enabled)
      self.patience = patience
      self.verbose = early_stop_verbose
      self.delta = early_stop_delta
      self.fold_num = fold_num
      self.exp_name = exp_name
      self.model_name = model_name
      self.loss_function_name=loss_function_name
      self.initial_lr=initial_lr
      self.final_lr=final_lr
      self.batch_size=batch_size
      if self.scheduler_name=='multisteplr':
        self.scheduler=MultiStepLR(self.optimizer,[5,15,25],0.1)
      elif self.scheduler_name=='reducelronplateau':
        self.scheduler=ReduceLROnPlateau(self.optimizer, mode='min',
        factor=0.1, patience=5, threshold=0.01,verbose=True)
      elif self.scheduler_name=='cosineannealinglr':
        self.scheduler=CosineAnnealingLR(self.optimizer,
          T_max=self.max_epochs,eta_min=float(self.optimizer.defaults['lr'])*10e-3, verbose=True)
      else:
        self.scheduler=None
      self.amp=amp
    
    def start(self):
        """
        Starts the training process.
        """
        # Training loop
        for epoch in range(self.max_epochs):
            self.epoch = epoch + 1
            train_loss = self.train()
            val_loss,metric_results = self.validation()
            logs={
                  'epoch': self.epoch,
                  'train_step':self.train_step,
                  'epoch_train_loss': train_loss,
                  'val_loss': val_loss,
                    }
            logs.update(metric_results)
            self.wandb.log(logs)
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                lr=self.scheduler.get_last_lr()
                self.wandb.log({
                    'epoch':self.epoch,
                    'lr':lr[0],
                })

            if self.early_stop_callback:
                self.early_stop_check(input_value=val_loss, loss=True, model=self.model, epoch=epoch)

            if self.early_stop:
                print("Early stopping")
                break

        self.reset_trainer()
        self.wandb.finish()
