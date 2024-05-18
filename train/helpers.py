import os
from torch import save
import numpy as np
import wandb

class Helpers:    
    def reset_trainer(self):
      """
        Resets the training state variables to their initial values.
      """
      # Reset training state variables
      self.best_r2s = -1
      self.best_r2s_epoch = -1
      self.val_loss_min = np.Inf
      self.best_score = None
      self.counter = 0
      self.early_stop = False
      self.epoch=0
      self.train_step=0
      self.val_step=0
    def early_stop_check(self, input_value, loss, model, epoch):
      """
      Checks if training should be stopped early based on validation performance.

      Parameters:
      ----------
      input_value : float
          The value to check for early stopping (e.g., validation loss).
      loss : bool
          Whether the input value represents a loss (True) or a score (False).
      model : torch.nn.Module
          The model being trained.
      epoch : int
          The current epoch number.
      """
      # Early stopping logic
      score=input_value
      if loss:
        score=-input_value
      if self.best_score is None:
        self.best_score=score
        self.best_epoch=epoch
        if self.save_best_model:
          self.save_model()
      elif (score + self.delta) < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # Stop training if validation loss does not improve for patience number of epochs
            if self.counter >= self.patience:
                self.early_stop = True
                if self.save_best_model:
                  self.save_model(name='last')
      else:
          self.best_score=score
          self.best_epoch=epoch
          self.counter = 0
          if self.save_best_model:
            self.save_model()
    def metric_calculation(self, y_pred, y):
        """
        Calculates and logs metrics based on predictions and actual values.

        Parameters:
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y : torch.Tensor
            The actual values.
        """
        # Metric calculation
        metric_results = self.metric_calculator.aggregate(y_pred.detach().cpu(), y.detach().cpu())
        r2s = metric_results['r2s']

        if r2s > self.best_r2s:
            self.best_r2s = r2s
            self.best_r2s_epoch = self.epoch
        print(f"current epoch: {self.epoch} current R2S: {r2s:.4f} Best R2S: {self.best_r2s:.4f} at epoch: {self.best_r2s_epoch}")
        return metric_results
    def save_model(self, name='best'):
        """
        Saves the model checkpoint.

        Parameters:
        ----------
        name : str, optional
            The name for the saved model (default is 'best').
        """
        # Save model checkpoint
        model_name = f"model_{self.exp_name}_{name}.pt"
        model_path = os.path.join(os.getcwd(), model_name)
        save(self.model.state_dict(), model_path)
        artifact = wandb.Artifact(name=model_name[:-3], type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        print(f"Model saved to {model_path}")