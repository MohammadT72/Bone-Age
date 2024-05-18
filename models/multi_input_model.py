# Pytorch
from torch import nn, no_grad, cat


class MultiInputModel(nn.Module):
    """
    A PyTorch neural network module that combines image features with extra features for classification.

    Attributes:
    ----------
    feature_extractor : torch.nn.Sequential
        The base model for feature extraction.
    fc1 : torch.nn.Linear
        The first fully connected layer.
    act1 : torch.nn.Module
        The activation function.
    drop1 : torch.nn.Dropout
        The dropout layer.
    fc2 : torch.nn.Linear
        The second fully connected layer.
    """
    def __init__(self, base_model, extra_info_dim, activation, drop_out, classifier_in_features, head_params, num_output):
        """
        Initializes the CombinedModel class.

        Parameters:
        ----------
        base_model : torch.nn.Module
            The base model for feature extraction.
        extra_info_dim : int
            The dimension of extra features to be concatenated with image features.
        activation : torch.nn.Module
            The activation function to be used.
        drop_out : float
            The dropout rate.
        classifier_in_features : int
            The number of input features for the classifier.
        head_params : int
            The number of parameters in the head layer.
        num_output : int
            The number of output features.
        """
        super(MultiInputModel, self).__init__()
        self.feature_extractor = base_model
        # Remove the last layer of the base model
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=classifier_in_features + extra_info_dim, out_features=head_params)
        self.act1 = activation
        self.drop1 = nn.Dropout(drop_out)
        self.fc2 = nn.Linear(in_features=head_params, out_features=num_output, bias=False)

    def forward(self, images, extra_info):
        """
        Forward pass of the CombinedModel.

        Parameters:
        ----------
        images : torch.Tensor
            The input image tensor.
        extra_info : torch.Tensor
            The extra features tensor.

        Returns:
        -------
        torch.Tensor
            The output of the model.
        """
        # Extract features from the images using the base model
        with no_grad():
            image_features = self.feature_extractor(images)
            image_features = image_features.view(image_features.size(0), -1)
        # Concatenate image features and extra features
        combined_features = cat((image_features, extra_info), dim=1)
        # Pass the combined features through the fully connected layers
        x = self.fc1(combined_features)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x