from torch import no_grad,cuda,squeeze,cat,tensor,float32,long
from print_color import print
class Validation:
    def validation(self):
        """
        Executes the validation loop.

        Returns:
        -------
        float
            The average validation loss for the epoch.
        """
        # Validation loop
        if (self.epoch) % self.val_interval == 0:
            self.model.eval()
            val_epoch_loss = 0
            self.val_step = 0
            y_pred = tensor([], dtype=float32, device=self.device)
            y = tensor([], dtype=long, device=self.device)

            with no_grad():
                for val_data in self.val_loader:
                    self.val_step += 1
                    val_images, val_labels = val_data[self.image_key].to(self.device), val_data[self.target_col_name].to(self.device)

                    if self.combined:
                        feature_tensors = [val_data[feature_name].unsqueeze(1).to(self.device) for feature_name in self.extra_info_list]
                        combined_features = cat(feature_tensors, dim=1)

                    with cuda.amp.autocast(enabled=self.enabled):
                        if self.combined:
                            val_outputs = self.model(val_images, combined_features)
                        else:
                            val_outputs = self.model(val_images)
                        val_outputs = squeeze(val_outputs, -1)
                    val_loss = self.loss_function(val_outputs, val_labels)
                    val_epoch_loss += val_loss.item()
                    y_pred = cat([y_pred, val_outputs], dim=0)
                    y = cat([y, val_labels], dim=0)

            val_epoch_loss /= self.val_step
            print(f"epoch {self.epoch} average val loss: {val_epoch_loss:.4f}", tag='success', tag_color='green', color='yellow')
            metric_results=self.metric_calculation(y_pred, y)
            return val_epoch_loss,metric_results