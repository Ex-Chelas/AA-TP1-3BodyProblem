import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load and preprocess the dataset
def load_and_preprocess(filename):
    df = pd.read_csv(filename)

    # Drop unnecessary columns and rows with all zeros (collisions)
    df_clean = df.drop(columns=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3', 'Id'])
    df_clean = df_clean[(df_clean != 0).any(axis=1)]

    # Create a new column 'tj_id' for trajectory IDs
    df_clean['tj_id'] = (df_clean['t'] == 0)
    df_clean['tj_id'] = df_clean['tj_id'].cumsum()

    return df_clean


# Function to split dataset into chunks by trajectory
def split_by_trajectories(df, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1"

    # Get unique trajectory IDs
    unique_tj_ids = df['tj_id'].unique()
    random.shuffle(unique_tj_ids)  # Shuffle to randomize

    num_trajectories = len(unique_tj_ids)
    train_size = int(train_frac * num_trajectories)
    val_size = int(val_frac * num_trajectories)

    # Split based on trajectory IDs
    train_ids = unique_tj_ids[:train_size]
    val_ids = unique_tj_ids[train_size:train_size + val_size]
    test_ids = unique_tj_ids[train_size + val_size:]

    train_data = df[df['tj_id'].isin(train_ids)]
    val_data = df[df['tj_id'].isin(val_ids)]
    test_data = df[df['tj_id'].isin(test_ids)]

    return train_data, val_data, test_data


# Function to clean and split the dataset
def process_and_store_splits(filename, train_frac=0.4, val_frac=0.3, test_frac=0.3):
    # Load and clean data
    df = load_and_preprocess(filename)

    # Split into train, val, and test based on trajectories
    train_data, val_data, test_data = split_by_trajectories(df, train_frac, val_frac, test_frac)

    # Store the clean data
    train_data.to_csv('train_data_clean.csv', index=False)
    val_data.to_csv('val_data_clean.csv', index=False)
    test_data.to_csv('test_data_clean.csv', index=False)

    return train_data, val_data, test_data


# Function to prepare the train X replicate initial positions across the trajectory, eliminating the actual positions.
def prepare_supervised_x_dataset(df):
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # For each unique trajectory
    for tj_id in df_copy['tj_id'].unique():
        # Extract the rows corresponding to the current trajectory
        trajectory_df = df_copy[df_copy['tj_id'] == tj_id]

        # Get the initial positions where t == 0 (first row of the trajectory)
        initial_positions = trajectory_df[trajectory_df['t'] == 0][['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].iloc[0]

        # Set all position columns in this trajectory to the initial positions
        df_copy.loc[df_copy['tj_id'] == tj_id, ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']] = initial_positions.values

    # Drop the 'tj_id' column as it's no longer necessary
    df_copy = df_copy.drop(columns=['tj_id'])
    # Remove the line of t=0 because it adds nothing to the prediction
    df_copy = df_copy[df_copy['t'] != 0]

    return df_copy


def prepare_supervised_y_dataset(df):
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    df_copy = df_copy[df_copy['t'] != 0]
    df_copy = df_copy[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]

    return df_copy.values


# Feature engineering functions
def add_time_features(df):
    df['t_squared'] = df['t'] ** 2
    df['t_cubed'] = df['t'] ** 3
    df['inv_t'] = 1 / (df['t'] + 1e-5)
    return df

def add_distance_features(df):
    df['dist_12'] = np.sqrt((df['x_1'] - df['x_2'])**2 + (df['y_1'] - df['y_2'])**2)
    df['dist_13'] = np.sqrt((df['x_1'] - df['x_3'])**2 + (df['y_1'] - df['y_3'])**2)
    df['dist_23'] = np.sqrt((df['x_2'] - df['x_3'])**2 + (df['y_2'] - df['y_3'])**2)
    return df

def add_squared_distance_features(df):
    df['squared_dist_12'] = df['dist_12'] ** 2
    df['squared_dist_13'] = df['dist_13'] ** 2
    df['squared_dist_23'] = df['dist_23'] ** 2
    return df

def add_inverse_distance_features(df):
    df['inv_dist_12'] = 1 / (df['dist_12'] + 1e-5)
    df['inv_dist_13'] = 1 / (df['dist_13'] + 1e-5)
    df['inv_dist_23'] = 1 / (df['dist_23'] + 1e-5)
    return df

def add_ratio_of_distance_features(df):
    df['ratio_dist_12_13'] = df['dist_12'] / (df['dist_13'] + 1e-5)
    df['ratio_dist_12_23'] = df['dist_12'] / (df['dist_23'] + 1e-5)
    return df


def prepare_kaggle_test_dataset(df):
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # Remove id column
    df_copy = df_copy.drop(columns=['Id'])

    # Rename the columns to match the expected format
    df_copy.columns = ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

    return df_copy


# Function to plot actual vs. predicted values
def plot_y_yhat(y_val, y_pred, plot_title="plot"):
    labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    _max = 500
    if len(y_val) > _max:
        idx = np.random.choice(len(y_val), _max, replace=False)
    else:
        idx = np.arange(len(y_val))

    plt.figure(figsize=(10, 10))
    for i in range(6):
        x0 = np.min(y_val[idx, i])
        x1 = np.max(y_val[idx, i])
        plt.subplot(3, 2, i + 1)
        plt.scatter(y_val[idx, i], y_pred[idx, i])
        plt.xlabel('True ' + labels[i])
        plt.ylabel('Predicted ' + labels[i])
        plt.plot([x0, x1], [x0, x1], color='red')
        plt.axis('square')

    plt.savefig(plot_title + '.pdf')
    # plt.show()
