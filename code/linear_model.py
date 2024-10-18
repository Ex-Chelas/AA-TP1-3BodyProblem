import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


# random.seed(42) Used for reproducibility on tests
# np.random.seed(42)

# Load and preprocess the dataset
def load_and_preprocess(filename):
    df = pd.read_csv(filename)

    # Drop unnecessary columns and rows with all zeros (collisions)
    df_clean = df.drop(columns=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3', 'Id'])
    df_clean = df_clean[(df_clean != 0).any(axis=1)]

    return df_clean


# Function to identify chunks (trajectories) and assign a trajectory ID
def assign_trajectory_ids(df):
    # Create a new column 'tj_id' for trajectory IDs
    df['tj_id'] = (df['t'] == 0) & (df.drop(columns=['t']) != 0).any(axis=1)
    df['tj_id'] = df['tj_id'].cumsum()  # Cumulative sum to assign trajectory numbers

    return df


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
def process_and_store_splits(filename, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    # Load and clean data
    df = load_and_preprocess(filename)

    # Assign trajectory IDs
    df = assign_trajectory_ids(df)

    # Split into train, val, and test based on trajectories
    train_data, val_data, test_data = split_by_trajectories(df, train_frac, val_frac, test_frac)

    # Store the clean data
    train_data.to_csv('train_data_clean.csv', index=False)
    val_data.to_csv('val_data_clean.csv', index=False)
    test_data.to_csv('test_data_clean.csv', index=False)

    return train_data, val_data, test_data


# Function to prepare the dataset (replicate initial positions across the trajectory)
def prepare_dataset(df):
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # For each unique trajectory (based on 'tj_id')
    for tj_id in df_copy['tj_id'].unique():
        # Extract the rows corresponding to the current trajectory
        trajectory_df = df_copy[df_copy['tj_id'] == tj_id]

        # Get the initial positions where t == 0 (first row of the trajectory)
        initial_positions = trajectory_df[trajectory_df['t'] == 0][['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].iloc[0]

        # Set all position columns in this trajectory to the initial positions
        df_copy.loc[df_copy['tj_id'] == tj_id, ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']] = initial_positions.values

    # Drop the 'tj_id' column as it's no longer necessary
    df_copy = df_copy.drop(columns=['tj_id'])

    return df_copy


# Function to build and train the linear regression model
def build_and_train_model(x_train, y_train):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardization
        ('regressor', LinearRegression())  # Linear regression model
    ])
    model_pipeline.fit(x_train, y_train)
    return model_pipeline


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
    plt.show()


# Function to rename columns of the test dataset from x0_1 to x_1
def rename_test_columns(df):
    df.rename(columns={'x0_1': 'x_1', 'y0_1': 'y_1', 'x0_2': 'x_2', 'y0_2': 'y_2', 'x0_3': 'x_3', 'y0_3': 'y_3'},
              inplace=True)
    return df


if __name__ == "__main__":
    # Load, process, and split the dataset
    process_and_store_splits('X_train.csv')

    train_data = pd.read_csv('train_data_clean.csv')
    val_data = pd.read_csv('val_data_clean.csv')
    # test_data = pd.read_csv('test_data_clean.csv') TODO: ONLY USE THIS ON FINAL VERSION, shall remain untouched

    # Prepare the inputs for training and predictions (dropping unnecessary columns)
    x_train = prepare_dataset(train_data)
    y_train = train_data[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]

    x_val = prepare_dataset(val_data)
    y_val = val_data[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]

    # Build and train the model
    model = build_and_train_model(x_train, y_train)

    # Train and validation predictions
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    # Calculate MSE for training and validation sets
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    print(f"Training MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")

    # Load test data and prepare it for prediction
    test_data = pd.read_csv("X_test.csv")
    test_data = test_data.drop(columns=['Id'])
    test_data = rename_test_columns(test_data)

    # Predict on the test data
    y_test_pred = model.predict(test_data)

    # Convert the predictions to a DataFrame and store it as a CSV
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_test_pred_df.insert(0, 'Id', y_test_pred_df.index)
    y_test_pred_df.to_csv('baseline-model.csv', index=False)
    print(f"Predictions saved to baseline-model.csv")
    print(f"File size: {y_test_pred_df.shape}")
