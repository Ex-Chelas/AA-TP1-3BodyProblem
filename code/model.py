import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import random
import matplotlib.pyplot as plt
import random
import numpy as np

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
    random.shuffle(unique_tj_ids) # Shuffle to randomize
    
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
    
    # Split into train, val, and test based on trajectories
    train_data, val_data, test_data = split_by_trajectories(df, train_frac, val_frac, test_frac)

    # Store the clean data
    train_data.to_csv('train_data_clean.csv', index=False)
    val_data.to_csv('val_data_clean.csv', index=False)
    test_data.to_csv('test_data_clean.csv', index=False)
    
    return train_data, val_data, test_data

# Function to prepare the train X (replicate initial positions across the trajectory, eliminating the actual positions.
def prepare_supervised_X_dataset(df):
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
    
    # Drop the 'tj_id' column as it's no longer needed
    df_copy = df_copy.drop(columns=['tj_id'])
    # Remove the line of t=0 because it adds nothing to the prediction
    df_copy = df_copy[df_copy['t'] != 0]

    return df_copy

def prepare_supervised_Y_dataset(df):
     # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    df_copy = df_copy[df_copy['t'] != 0]
    df_copy = df_copy[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]
    
    
    return df_copy.values
    

def prepare_kaggle_test_dataset(df):
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # Remove id column
    df_copy = df_copy.drop(columns=['Id'])
    
    # Rename the columns to match the expected format
    df_copy.columns = ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    
    return df_copy

# Function to build and train the linear regression model
def build_and_train_model(X_train, y_train):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardization
        ('regressor', LinearRegression())  # Linear regression model
    ])
    
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

# Function to plot actual vs predicted values
def plot_y_yhat(y_val, y_pred, plot_title="plot"):
    labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    MAX = 500
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val), MAX, replace=False)
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

# Main script
def main():
    # Load, process, and split the dataset
    # process_and_store_splits('X_train.csv')

    train_data = pd.read_csv('train_data_clean.csv')
    val_data = pd.read_csv('val_data_clean.csv')
    #test_data = pd.read_csv('test_data_clean.csv') TODO: ONLY USE THIS ON FINAL VERSION, shall remain untouched

    # Prepare the inputs for training and predictions (dropping unnecessary columns)
    X_train = prepare_supervised_X_dataset(train_data)
    y_train = prepare_supervised_Y_dataset(train_data)

    X_val = prepare_supervised_X_dataset(val_data)
    y_val = prepare_supervised_Y_dataset(val_data)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Train and validation predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate MSE for training and validation sets
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    print(f"Training MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")

    # plot_y_yhat(y_train, y_train_pred, 'Train Predicted vs Expected')
    plot_y_yhat(y_val, y_val_pred, 'Validation Predicted vs Expected')

    # Load test data and prepare it for prediction
    test_data = pd.read_csv("X_test.csv")
    clean_test_data = test_data.drop(columns=['Id'])
    
    # Change columns to follow the format
    clean_test_data.columns = ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

    # Predict on the test data
    y_test_pred = model.predict(clean_test_data)

    # Convert the predictions to a DataFrame and store it as a CSV
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_test_pred_df.insert(0, 'Id', y_test_pred_df.index)
    
    y_test_pred_df.to_csv('baseline-model.csv', index=False)

if __name__ == "__main__":
    main()