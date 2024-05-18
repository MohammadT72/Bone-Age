# Bone Age Prediction - GitHub Project

## Overview
This project is aimed at predicting bone age from medical images using a deep learning model built with PyTorch and MONAI. The project includes data preprocessing, model training, and evaluation using k-fold cross-validation. We leverage multiple libraries, including Albumentations for data augmentation, Weights and Biases (WandB) for experiment tracking, and Sklearn for stratified k-fold splitting.

## Project Structure
The main components of this project include:
1. **Data Preprocessing**: Scripts to load and preprocess the images and corresponding labels.
2. **Model Training**: Implementation of a custom trainer that supports k-fold cross-validation and advanced features like early stopping and mixed precision training.
3. **Evaluation**: Methods to compute and log metrics, ensuring that the model's performance is thoroughly evaluated.

## Dependencies
To install the required libraries, run:
```bash
pip install -r requirements.txt
```
### Libraries used:
- **Standard Libraries**: os, time, glob, zipfile, yaml, random
- **Data Manipulation and Visualization**: pandas, numpy, matplotlib, cv2, PIL
- **Deep Learning**: torch, torchvision
- **Data Augmentation**: albumentations, MONAI
- **Metrics**: torcheval, torchmetrics
- **Experiment Tracking**: wandb

## Configuration
The project is configurable via a YAML file. An example configuration is provided in `config_test.yaml`. Key configurations include model architecture, training parameters, and data preprocessing steps.

## Code Overview

### Data Preparation
You can download the dataset using the following commands:
```
wget https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set.zip
wget https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Training+Set+Annotations.zip
wget https://s3.amazonaws.com/east1.public.rsna.org/AI/2017/Bone+Age+Validation+Set.zip

```
You should create a data.csv file similar to the example provided. Below is a script to help you prepare the data:
```
import pandas as pd
import glob
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# Read the CSV file containing data statistics into a DataFrame
df = pd.read_csv('/content/results/stats.csv', index_col=0)

# Get a list of object detection txt file paths
txt_path = glob.glob('/content/txt_files/*.txt')

# Iterate through the combined list of text file paths
df['mask'] = None
for txt in txt_path:
    # Extract the ID from the filename
    id = int(os.path.basename(txt).split('.txt')[0])
    # Find the index of the DataFrame row where the 'id' column matches the extracted ID
    index = df[df.id == id].index[0]
    # Set the 'mask' column value at the found index to the current text file path
    df.loc[index, 'mask'] = txt

# Normalize the 'boneage'
df['boneage_normalized'] = (df['boneage'] - df['boneage'].mean()) / df['boneage'].std()
# Convert the 'male' column from boolean to integer (True -> 1, False -> 0)
df['male'] = df['male'].astype(int)
df['boneage_decile'] = pd.cut(df['boneage'], bins=10, labels=False)
df['mean_pixel_decile'] = pd.cut(df['mean_pixel_value'], bins=10, labels=False)
df['boneage_decile_label'] = pd.cut(df['boneage'], bins=10)
df['mean_pixel_decile_label'] = pd.cut(df['mean_pixel_value'], bins=10)

def StratifiedGroupKFoldWrapper(main_df, target_col='boneage_decile',n_splits=10):
    """
    Splits the DataFrame into stratified train, validation, and test sets.

    Parameters:
    main_df (pd.DataFrame): The main DataFrame to be split.
    target_col (str): The target column for stratification.
    n_splits (int): The number of splits for KFold.

    Returns:
    pd.DataFrame: The DataFrame with new columns indicating the fold assignments.
    """

    # Reset the index of the DataFrame
    data = main_df.reset_index(drop=True)
    # Initialize StratifiedKFold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    data['test'] = False

    # Perform the first split to create the test set
    for fold, (train_val_idx, test_idx) in enumerate(cv.split(data, data[target_col])):
        data.loc[test_idx, 'test'] = True
        break

    # Initialize a new StratifiedKFold for train/val splits
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=26)
    train_val = data[data['test'] == False].reset_index()

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_val, train_val[target_col])):
        data[f'fold_{fold+1}'] = None
        old_index_train = train_val['index'].iloc[train_idx]
        old_index_val = train_val['index'].iloc[val_idx]
        data.loc[old_index_train, f'fold_{fold+1}'] = 'train'
        data.loc[old_index_val, f'fold_{fold+1}'] = 'val'

    return data

new_df = StratifiedGroupKFoldWrapper(df)
new_df.to_csv('/content/data.csv')
```
### Imports and Setup
The project imports a variety of libraries required for data handling, model building, training, and evaluation. It also sets up a deterministic environment to ensure reproducibility.

### Data Preprocessing
Data is preprocessed using custom classes and transforms defined to handle image normalization, cropping, and augmentation. The data is split into train, validation, and test sets using stratified k-fold splitting to maintain class balance.

### Model Training
A `TrainerWrapper` class encapsulates the training process, including:
- Initializing data loaders, model, optimizer, and loss function
- Implementing a training loop with support for mixed precision training and WandB logging
- Early stopping based on validation performance

### Metrics Calculation
Metrics such as Mean Squared Error (MSE), R2 Score, and Mean Absolute Error (MAE) are calculated using the `MetricsCalculator` class.

### K-Fold Cross-Validation
The `KFoldTrainer` class manages the k-fold cross-validation, allowing the model to be trained and evaluated across multiple folds. This class ensures that the data is split correctly and that the model's performance is averaged over several runs.

## Usage

### Running the Project

1. **Prepare Data**: Ensure your data is correctly formatted and paths are correctly specified in the configuration file.
2. **Configure Parameters**: Edit the `config_test.yaml` file to set your experiment parameters.

### K-Fold Cross-Validation

To start the training process with k-fold cross-validation, run:
```bash
python final_training.py --config_path=config.yaml
```

### Final Training

To train on the full dataset after k-fold validation, run:
```bash
python final_training.py --config_path=config.yaml
```

## Example Code
Here's an example of how to initialize and run the k-fold trainer:
```python
data = new_df[["id","boneage","boneage_normalized","male","path","mask","test"] + [f'fold_{x}' for x in range(1,11)]]
data = data[data['mask'].notnull() & data['path'].notnull()]
data = data.reset_index(drop=True)

kfoldtrainer = KFoldTrainer(
    data=data,
    frac=1.0,
    cache_rate=0.3,
    combined=True,
    extra_info_list=['male'],
    config_path='/content/drive/MyDrive/data/config_test.yaml',
    exp_name='bone_age_01',
    run=1,
    user='ai_medvira',
    metrics=['mse', 'r2s', 'mae'],
    pretrained_model_name=None,
)
kfoldtrainer.k_fold()
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.