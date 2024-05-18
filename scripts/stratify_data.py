import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import argparse

def preprocess_data(df):
    df['boneage_decile'] = pd.cut(df['boneage'], bins=10, labels=False)
    df['mean_pixel_decile'] = pd.cut(df['mean_pixel_value'], bins=10, labels=False)
    df['boneage_decile_label'] = pd.cut(df['boneage'], bins=10)
    df['mean_pixel_decile_label'] = pd.cut(df['mean_pixel_value'], bins=10)
    return df

def StratifiedGroupKFoldWrapper(main_df, target_col='boneage_decile',
                                group_col='mean_pixel_bins', group=False, n_splits=10):
    """
    Splits the DataFrame into stratified train, validation, and test sets.

    Parameters:
    main_df (pd.DataFrame): The main DataFrame to be split.
    target_col (str): The target column for stratification.
    group_col (str): The column for group-wise splitting.
    group (bool): Whether to use StratifiedGroupKFold.
    n_splits (int): The number of splits for KFold.

    Returns:
    pd.DataFrame: The DataFrame with new columns indicating the fold assignments.
    """

    # Reset the index of the DataFrame
    data = main_df.reset_index(drop=True)

    if group:
        # Initialize StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        data['test'] = False

        # Perform the first split to create the test set
        for fold, (train_val_idx, test_idx) in enumerate(cv.split(data, data[target_col], data[group_col])):
            data.loc[test_idx, 'test'] = True
            break

        # Initialize a new StratifiedGroupKFold for train/val splits
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=26)
        train_val = data[data['test'] == False].reset_index()

        for fold, (train_idx, val_idx) in enumerate(cv.split(train_val, train_val[target_col], train_val[group_col])):
            data[f'fold_{fold+1}'] = None
            old_index_train = train_val['index'].iloc[train_idx]
            old_index_val = train_val['index'].iloc[val_idx]
            data.loc[old_index_train, f'fold_{fold+1}'] = 'train'
            data.loc[old_index_val, f'fold_{fold+1}'] = 'val'

    else:
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

def main(input_csv, output_csv):
    # Load the data
    df = pd.read_csv(input_csv)

    # Preprocess the data
    df = preprocess_data(df)

    # Perform stratified splitting
    stratified_df = StratifiedGroupKFoldWrapper(df)

    # Save the stratified DataFrame to a new CSV file
    stratified_df.to_csv(output_csv, index=False)
    print(f"Stratified data saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a CSV file, stratify it, and save to another CSV file.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file.')

    args = parser.parse_args()
    main(args.input_csv, args.output_csv)
