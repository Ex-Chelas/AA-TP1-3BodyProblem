from idlelib.autocomplete import TRY_A

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

random.seed(42)
np.random.seed(42)


# Function to shuffle and split data indices based on a given chunk size
def get_split_indices(dataset, train_frac, val_frac, test_frac, chunk_len=256):
    num_trajectories = len(dataset) // chunk_len  # Calculate number of full chunks
    assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1"

    indices = np.arange(0, num_trajectories * chunk_len, chunk_len)  # Generate chunk starting indices
    random.shuffle(indices)  # Shuffle the indices to randomize

    train_size = int(train_frac * num_trajectories)
    val_size = int(val_frac * num_trajectories)

    train_idxs = indices[:train_size]
    val_idxs = indices[train_size:train_size + val_size]
    test_idxs = indices[train_size + val_size:]

    return train_idxs, val_idxs, test_idxs

# Function to clean out the rows with all zero values (colisions)
def clean_data(dataset_matrix):
    return np.array([row for row in dataset_matrix if not (row[-1] != 0 and all(val == 0.0 for val in row[:-1]))])

# Function to prepare the input dataset by replicating the first row across the trajectory
def prepare_dataset(matrix):
    first_entry = []
    processed_matrix = []

    for row in matrix:
        temp_row = row.copy()

        if row[7] % 256 == 0:  # Identify trajectory start
            first_entry = row.copy()
        else:
            for i in range(1, 7):
                if row[-1] != 0 and all(val == 0.0 for val in row[:-1]):
                    continue
                temp_row[i] = first_entry[i]

        processed_matrix.append(np.delete(temp_row, [7]))  # Remove 'id' column

    return np.array(processed_matrix)

def prepare_test_dataset(matrix):
    """
    Prepare the test dataset using the initial positions and the desired time steps.

    Parameters:
    - matrix: A numpy array with columns [Id, t, x0_1, y0_1, x0_2, y0_2, x0_3, y0_3]

    Returns:
    - processed_matrix: A numpy array ready for model input
    """
    processed_matrix = []
    first_entry = None

    for row in matrix:
        temp_row = row.copy()

        # Identify trajectory start
        if row[0] % 256 == 0:  # 'Id' is at index 0
            first_entry = row.copy()

        # Replicate initial positions (from indices 2 to 7)
        temp_row[2:8] = first_entry[2:8]

        # Remove 'Id' column (index 0)
        temp_row = np.delete(temp_row, [0])

        processed_matrix.append(temp_row)

    return np.array(processed_matrix)



# Load and preprocess the dataset
def load_and_preprocess(filename):
    raw_data = np.loadtxt(filename, delimiter=",", skiprows=1)
    clean_data = np.delete(raw_data, [3, 4, 7, 8, 11, 12], axis=1)  # Remove unnecessary columns
    return clean_data

# Split the data into training, validation, and test sets, and clean the data
def process_splits(data, train_ratio, val_ratio, test_ratio):
    train_idxs, val_idxs, test_idxs = get_split_indices(data, train_ratio, val_ratio, test_ratio)

    train_data = np.concatenate([data[idx: idx + 256] for idx in train_idxs], axis=0)
    val_data = np.concatenate([data[idx: idx + 256] for idx in val_idxs], axis=0)
    test_data = np.concatenate([data[idx: idx + 256] for idx in test_idxs], axis=0)

    print(f"Training data size before cleaning: {train_data.shape}")
    print(f"Validation data size before cleaning: {val_data.shape}")

    train_clean = clean_data(train_data)
    val_clean = clean_data(val_data)
    test_clean = clean_data(test_data)

    print(f"Training data size after cleaning: {train_clean.shape}")
    print(f"Validation data size after cleaning: {val_clean.shape}")

    return train_clean, val_clean, test_clean

def analyze_distributions(train_data, val_data):
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    print("Training Data Description:")
    print(train_df.describe())

    print("Validation Data Description:")
    print(val_df.describe())


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



#depois eu faço bonito é python chill
training_MSE = []
validation_MSE = []
test_MSE = []
current = 99

def do_the_thing(data):
    train_data, val_data, test_data = process_splits(data, 0.7, 0.15, 0.15)

    # Prepare the inputs for training and predictions
    X_train = prepare_dataset(train_data)
    X_val = prepare_dataset(val_data)
    X_test = prepare_dataset(test_data)

    y_train = np.delete(train_data, [0, 7], axis=1)  # Delete time and id columns
    y_val = np.delete(val_data, [0, 7], axis=1)
    y_test = np.delete(test_data, [0, 7], axis=1)

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

    training_MSE.append(train_mse)
    validation_MSE.append(val_mse)
    # Plot the actual vs predicted graph for training set
    # plot_y_yhat(y_train, y_train_pred, plot_title="Training Set: Actual vs Predicted")

    # Test predictions and MSE calculation
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Test MSE: {test_mse}")
    test_MSE.append(test_mse)

    global current
    current = (test_mse + train_mse + val_mse) / 3

    return model

# Main script
def main():
    # Load the dataset
    data = load_and_preprocess("../data/X_train.csv")
    test_data = np.loadtxt("../data/X_test.csv", delimiter=",", skiprows=1)
    best = 100
    best_model = None
    for i in range(1):
        model = do_the_thing(data)
        if current < best:
            best = current
            best_model = model

    x_teste = prepare_test_dataset(test_data)
    y_teste_pred = best_model.predict(x_teste)
    y_teste_pred_df = pd.DataFrame(y_teste_pred)
    id_column = y_teste_pred[:, 0]  # Extract 'Id' from original test data
    y_teste_pred_df.insert(0, 'Id', id_column)
    y_teste_pred_df.to_csv('test_data_pred.csv', index=False)



    # Plot the actual vs predicted graph for test set
    #plot_y_yhat(y_test, y_test_pred, plot_title="Test Set: Actual vs Predicted")
    print(f"avg Training MSE: {np.mean(training_MSE)}")
    print(f"avg Validation MSE: {np.mean(validation_MSE)}")
    print(f"avg Test MSE: {np.mean(test_MSE)}")

# Run the main function
if __name__ == "__main__":
    main()
