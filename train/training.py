from torch import cuda,squeeze,cat

class Train:
    def train(self):
            """
            Executes the training loop for one epoch.

            Returns:
            -------
            float
                The average training loss for the epoch.
            """
            # Training loop
            print("-" * 10)
            print(f"epoch {self.epoch}/{self.max_epochs}")
            self.model.train()
            epoch_loss = 0
            self.train_step = 0

            for batch_data in self.train_loader:
                self.train_step += 1
                inputs, labels = batch_data[self.image_key].to(self.device), batch_data[self.target_col_name].to(self.device)

                if self.combined:
                    # For combined model which accepts both image and extra features as input
                    feature_tensors = [batch_data[feature_name].unsqueeze(1).to(self.device) for feature_name in self.extra_info_list]
                    combined_features = cat(feature_tensors, dim=1)

                self.optimizer.zero_grad()
                with cuda.amp.autocast(enabled=self.enabled):
                    if self.combined:
                        outputs = self.model(inputs, combined_features)
                    else:
                        outputs = self.model(inputs)
                    outputs = squeeze(outputs, -1)
                    loss = self.loss_function(outputs, labels)
                self.wandb.log({
                    'train_step':self.train_step,
                    'step_train_loss':loss.item()
                })
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += loss.item()

                if self.train_step % self.show_interval == 0:
                    print(f"{self.train_step}/{self.train_num_batch}, train_loss: {loss.item():.4f}")

            epoch_loss /= self.train_step
            print(f"epoch {self.epoch} average loss: {epoch_loss:.4f}")
            return epoch_loss