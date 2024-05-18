import pandas as pd
# Pytorch
from torch import nn
from torch import load
from torchvision import models as tv_models
from models.multi_input_model import MultiInputModel

class ModelSelection(object):
    def get_pretrained_model(self,):
      if self.config==None:
        self.read_config()
      self.prepare_settings()
      self.create_wandb_exp()
      if self.pretrained_model_name!=None:
        self.load_model_from_wandb()
      return self.model
    def check_model(self):
        """
        Checks the model parameters, their devices, and whether they require gradients.
        """
        names = []
        devices_list = []
        params_grad = []
        for name, param in self.model.named_parameters():
            names.append(name)
            devices_list.append(param.device)
            params_grad.append(param.requires_grad)
        # Create a DataFrame with the model parameters information
        params_df = pd.DataFrame.from_dict({'Names': names, 'Requires_grad': params_grad, 'devices': devices_list})
        # Print the number of devices used, parameters that require gradients, and their names
        print(f'Number of devices : {params_df.devices.value_counts()}')
        print(f'Number of Requires_grad : \n{params_df.Requires_grad.value_counts()}')
        print(f'List of True Requires_grad : \n{params_df[params_df.Requires_grad==True]}')

    def load_model_from_wandb(self,):
        """
        Load a PyTorch model state dictionary from Weights & Biases (wandb).
        """
        # Access and download the model. Returns the path to the downloaded artifact
        model_path = self.wandb.use_model(name=f"{self.pretrained_model_name}")
        # Load the state dictionary
        state_dict = load(model_path)
        # Initialize the model and load the state dictionary
        self.model.load_state_dict(state_dict)
    def replace_classifier(self):
        """
        Replaces the classifier of the model based on the specified settings.
        """
        # Define a dictionary of activation functions
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'Leakyrelu': nn.LeakyReLU(),
        }
        # Define a dictionary of classifiers
        classifiers = {
            'medium': nn.Sequential(
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=self.classifier_in_features, out_features=self.head_params),
                activations[self.activation],
                nn.Dropout(self.drop_out),
                nn.Linear(in_features=self.head_params, out_features=self.num_output, bias=False)
            ),
        }
        # If freeze_base is True, set requires_grad to False for all parameters in the model
        if self.freeze_base_first:
            for param in self.model.parameters():
                param.requires_grad = False
        # Replace the model with CombinedModel if combined is True
        if self.combined:
            self.model = MultiInputModel(self.model, self.extra_info_dim,
                                       activations[self.activation], self.drop_out,
                                       self.classifier_in_features,
                                       self.head_params, self.num_output)
        # Replace the classifier of the model with the new classifier based on the specified classifier key
        else:
            if self.classifier_key == 'classifier':
                self.model.classifier = classifiers['medium']
            elif self.classifier_key == 'medium_flatten':
                self.model.classifier = classifiers['medium_flatten']
            elif self.classifier_key == 'fc':
                self.model.fc = classifiers['medium']
            elif self.classifier_key == 'head':
                self.model.head = classifiers['medium']
            elif self.classifier_key == 'heads':
                self.model.heads = classifiers['medium']

    def model_selection(self):
        """
        Selects and returns a model based on the specified model name and transfer learning settings.
        """
        # Define a dictionary of available models
        models = {
            'vgg16': tv_models.vgg16,
            'vgg19': tv_models.vgg19,
            'densenet121': tv_models.densenet121,
            'densenet161': tv_models.densenet161,
            'densenet169': tv_models.densenet169,
            'densenet201': tv_models.densenet201,
            'efficientnet_b0': tv_models.efficientnet_b0,
            'efficientnet_b3': tv_models.efficientnet_b3,
            'efficientnet_b7': tv_models.efficientnet_b7,
            'efficientnet_v2_s': tv_models.efficientnet_v2_s,
            'efficientnet_v2_m': tv_models.efficientnet_v2_m,
            'mobilenet_v2': tv_models.mobilenet_v2,
            'resnet18': tv_models.resnet18,
            'resnet34': tv_models.resnet34,
            'resnet50': tv_models.resnet50,
            'resnet101': tv_models.resnet101,
            'resnet152': tv_models.resnet152,
            'wide_resnet50_2': tv_models.wide_resnet50_2,
            'inception_v3': tv_models.inception_v3,
            'vit_b_16': tv_models.vit_b_16,
            'maxvit_t': tv_models.maxvit_t,
            'swin_t': tv_models.swin_t,
            'swin_v2_t': tv_models.swin_v2_t,
        }
        # If the specified model name is not in the dictionary of available models, raise an error
        if self.model_name not in list(models.keys()):
            raise ValueError(f'model {self.model_name} is not in the implemented list')
        # Return the selected model with the specified weights
        if self.transfer_learning:
            return models[self.model_name](weights=self.weights_name)
        else:
            return models[self.model_name]()