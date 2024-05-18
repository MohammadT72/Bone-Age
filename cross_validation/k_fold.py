import os
import yaml
import wandb
import pandas as pd
from print_color import print

from torch import nn,cuda,device,optim

from data.data_loader import DataLoaderWrapper
from data.transforms import Transforms
from models.model_selection import ModelSelection
from train.trainer import TrainerWrapper
from metrics.metric_calculator import MetricsCalculator

class KFoldTrainer(ModelSelection,Transforms,
                   DataLoaderWrapper,TrainerWrapper,
                   MetricsCalculator):
  def __init__(self,config_path,
              data=None,
              exp_name='k-fold',
              run=0,
              pretrained_model_name=None,
              user='medvira_ai',
              image_key='image',
              mask_key='mask',
              target_col_name='target',
              cache_rate=1.0,
              save_best_model=True,
              combined=False,
              extra_info_list=[],
              metrics='all',
              k=10,
              frac=1.0
               ):
    super().__init__()
    self.data=data
    self.config_path=config_path
    self.exp_name=exp_name
    self.run=run
    self.user=user
    self.pretrained_model_name=pretrained_model_name
    self.image_key=image_key
    self.mask_key=mask_key
    self.cache_rate=cache_rate
    self.combined=combined
    self.extra_info_list=extra_info_list
    self.extra_info_dim=len(extra_info_list)
    self.target_col_name=target_col_name
    self.metrics=metrics
    self.save_best_model=save_best_model
    self.k=k
    self.frac=frac
    self.start_fold=0
    self.config=self.read_config()
  def read_config(self):
    # Read YAML file
    if not os.path.exists(self.config_path):
        raise ValueError("Config file does not exist: {}".format(self.config_path))
    with open(self.config_path, 'r') as stream:
        self.config = yaml.safe_load(stream)
    if isinstance(self.data,type(None)):
       self.image_key=self.config['image_key'][0]
       self.mask_key=self.config['mask_key'][0]
       path=self.config['data_path'][0]
       self.data=pd.read_csv(path,index_col=0)
       if self.data[self.image_key].isnull().any():
            raise ValueError(f"Column in Data '{self.image_key}' contains null values.")
       self.exp_name=self.config['exp_name'][0]
       self.run=self.config['run'][0]
       self.user=self.config['user'][0]
       self.pretrained_model_name=self.config['pretrained_model_name'][0]
       self.cache_rate=self.config['cache_rate'][0]
       self.combined=self.config['combined'][0]
       self.extra_info_list=self.config['extra_info_list']
       self.extra_info_dim=len(self.extra_info_list)
       self.target_col_name=self.config['target_col_name'][0]
       self.metrics=self.config['metrics']
       self.save_best_model=self.config['save_best_model'][0]
       self.k=self.config['k'][0]
       self.frac=self.config['frac'][0]
  def create_wandb_exp(self,fold=None):
    try:
      wandb.ensure_configured()
      logged_in = wandb.api.api_key is not None
    except Exception as e:
        logged_in = False

    if logged_in:
        print("You are logged in to WandB.", color='green')
    else:
        print("You are not logged in to WandB. Please log in.", color='red')
        wandb.login()
    config = {
      "model_name": self.model_name,
      "num_output": self.num_output,
      "augmentation": self.augmentation,
      "freeze_base_first": self.freeze_base_first,
      "transfer_learning": self.transfer_learning,
      "head_params": self.head_params,
      "drop_out": self.drop_out,
      "weights_name": self.weights_name,
      "dataset_name": self.dataset_name,
      "preprocessing_steps_train": self.preprocessing_steps_train,
      "preprocessing_steps_validation": self.preprocessing_steps_val,
      "batch_size": self.batch_size,
      "schedular_name": self.scheduler_name,
      "GPU": self.gpu,
      "AMP": self.amp,
      "epochs": self.max_epochs,
      "loss_function_name": self.loss_name,
      "optimizer_name": self.optimizer_name,
      "initial_lr":self.initial_lr,
      "final_lr":self.final_lr,
      "early_stop_callback":self.early_stop_callback,
    }
    if fold!=None:
      self.wandb=wandb.init(
                        project=self.exp_name,
                        entity=self.user,
                        name=f'run_{self.run}_fold_{fold+1}',
                        notes=f"fold {fold+1}",
                        config=config)
    else:
      self.wandb=wandb.init(
                        project=self.exp_name,
                        entity=self.user,
                        name=f'run_{self.run}',
                        notes="final_training",
                        config=config)

  def prepare_settings(self):
    # global settings
    self.gpu = self.config['gpu'][0]
    self.device=device('cpu')
    if self.gpu:
      self.device=device("cuda" if cuda.is_available() else "cpu")

    # Model settings
    self.model_name = list(self.config['models'].keys())[0]
    self.num_output = self.config['num_output'][0]
    classifier_size = self.config['classifier_size'][0]
    self.activation = self.config['activation_type'][0]
    self.augmentation = self.config['augmentations'][0]
    self.freeze_base_first = self.config['freeze_base_first'][0]
    self.transfer_learning = self.config['transfer_learning'][0]
    self.head_params = self.config['hidden_params'][0]
    self.drop_out= self.config['drop_out'][0]
    self.input_size = tuple(self.config['models'][self.model_name]['input_size'])
    self.weights_name = self.config['models'][self.model_name]['weights']
    self.classifier_key = self.config['models'][self.model_name]['classifier']['key']
    self.classifier_in_features = self.config['models'][self.model_name]['classifier']['in_features']
    self.model=self.model_selection()
    self.replace_classifier()
    self.model=self.model.to(self.device)

    # Transforms settings
    self.check = self.config['check'][0]
    self.dataset_name = self.config['dataset_name'][0]
    self.defualt_mode = self.config['defualt_mode'][0]
    self.make_transforms(self.config,self.model_name,self.augmentation)

    # Metric settings
    self.metric_calculator=self.make_metric_calculator()


    # Training settings
    self.initial_lr = self.config['initial_lr'][0]
    self.final_lr = self.config['final_lr'][0]
    self.scheduler_name = self.config['scheduler_name'][0]
    self.amp = self.config['amp'][0]
    self.val_interval = self.config['val_interval'][0]
    self.train_interval = self.config['train_interval'][0]
    self.early_stop_callback = self.config['early_stop_callback'][0]
    self.patience = self.config['patience'][0]
    self.early_stop_verbose = self.config['early_stop_verbose'][0]
    self.early_stop_delta = self.config['early_stop_delta'][0]
    self.max_epochs = self.config['max_epochs'][0]
    self.batch_size = self.config['batch_size'][0]
    self.optimizer_name = self.config['optimizer'][0]
    self.optimizer = self.get_optimizer(self.optimizer_name)(params=self.model.parameters(), lr=self.initial_lr)
    self.loss_name = self.config['losses'][0]
    self.loss_function = self.get_loss_function(self.loss_name)
    print(f'***** Experiment: {self.exp_name} *****')
    print(f'''Settings:
    --augmentations: {self.augmentation}
    --model: {self.model_name}
    --classifier_size: {classifier_size}
    --activation_type: {self.activation}
    --head_params: {self.head_params}
    --drop_out: {self.drop_out}
    --batch size: {self.batch_size}
    --loss_name: {self.loss_name}
    --optimizer: {self.optimizer_name}
    --initial learning rate: {self.initial_lr}
    --final learning rate: {self.final_lr}
    --weights: {self.weights_name}''')


  def get_loss_function(self,loss_func_name:str):
    if loss_func_name == 'mse':
      loss_function=nn.MSELoss()
    else:
      raise ValueError(f'{loss_func_name} is not in the implemented list')
    return loss_function
  def get_optimizer(self,opt_name:str):
    # Return an instance of the specified optimizer class
    if opt_name == 'adam':
        opt = optim.Adam
    elif opt_name == 'adamw':
        opt = optim.AdamW
    elif opt_name == 'rmsprop':
        opt = optim.RMSprop
    elif opt_name == 'sgd':
        opt = optim.SGD
    else:
      raise ValueError(f'{opt_name} is not in the implemented list')
    return opt
  def make_metric_calculator(self):
    # Define a list of available metrics
    selected_metrics=["r2s",'mse','mae']
    # If all metrics are requested, select all available metrics
    if not self.metrics=='all':
        selected_metrics=self.metrics
    # Return an instance of the metrics_calculator class with the selected metrics
    return MetricsCalculator(metrics=selected_metrics)
  def reset_fold(self):
    # Reset the training settings
    self.prepare_settings()
    # Resets the data loaders.
    self.train_data_list=None
    self.val_data_list=None
    self.test_data_list=None
    self.train_loader=None
    self.val_loader=None
  def k_fold(self,start_fold=None):
    """
    Performs k-fold cross-validation
    """
    if self.config==None:
      self.read_config()
    if start_fold!=None:
      self.start_fold=start_fold
    for fold in range(self.start_fold,self.k):
      # select the data for this fold.
      print(f'********* fold {fold+1} *********')
      self.reset_fold()
      self.select_data_list(fold,frac=self.frac)
      self.prepare_settings()
      self.make_dataloader(self.batch_size,self.dataset_name,cache_rate=self.cache_rate)
      self.create_wandb_exp(fold)
      if self.pretrained_model_name!=None:
        self.load_model_from_wandb()
      self.check_data(fold)
      self.check_model()
      self.reset_trainer()
      self.initialize_training_settings(
              self.train_loader,
              self.val_loader,
              self.model,self.optimizer,self.loss_function,
              self.max_epochs,
              self.target_col_name,
              self.image_key,
              self.batch_size,
              self.metric_calculator,
              self.exp_name,
              self.model_name,
              self.loss_name,
              self.initial_lr,
              self.final_lr,
              self.scheduler_name,
              self.amp,
              self.val_interval,
              self.device,
              self.train_interval,
              self.early_stop_callback,
              self.patience,
              self.early_stop_verbose,
              self.early_stop_delta,
              fold)
      self.start()
  def final_training(self,):
    """
    Performs final training with all training set, and the test set
    """
    if self.config==None:
      self.read_config()
    print(f'********* final training *********')
    self.select_data_list_final(frac=self.frac)
    self.prepare_settings()
    self.make_dataloader(self.batch_size,self.dataset_name,cache_rate=self.cache_rate)
    self.create_wandb_exp()
    if self.pretrained_model_name!=None:
        self.load_model_from_wandb()
    self.check_data('final_training')
    self.check_model()
    self.reset_trainer()
    self.initialize_training_settings(
            self.train_loader,
            self.val_loader,
            self.model,self.optimizer,self.loss_function,
            self.max_epochs,
            self.target_col_name,
            self.image_key,
            self.batch_size,
            self.metric_calculator,
            self.exp_name,
            self.model_name,
            self.loss_name,
            self.initial_lr,
            self.final_lr,
            self.scheduler_name,
            self.amp,
            self.val_interval,
            self.device,
            self.train_interval,
            self.early_stop_callback,
            self.patience,
            self.early_stop_verbose,
            self.early_stop_delta,
            'final_training')
    self.start()