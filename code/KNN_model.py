import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from model import prepare_supervised_x_dataset, prepare_supervised_y_dataset, plot_y_yhat, process_and_store_splits, add_time_features, add_distance_features

train_mse_list = []
val_mse_list = []
time_list = []
models = []


def plot_k_precision(train_mse_list, val_mse_list):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16), train_mse_list, label="Training MSE", marker='o')
    plt.plot(range(1, 16), val_mse_list, label="Validation MSE", marker='o')

    min_val_mse = min(val_mse_list)
    min_k = val_mse_list.index(min_val_mse) + 1
    plt.axhline(y=min_val_mse, color='r', linestyle='--', label=f'Lowest Validation MSE = {min_val_mse:.4f} at K:{min_k}')

    plt.title('MSE vs K Value for Training and Validation Sets')
    plt.xlabel('K Value')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_k_time(time_list):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16), time_list, label="Time Taken", marker='o', color='r')
    plt.title('Time Taken vs K Value for Training and Predicting')
    plt.xlabel('K Value')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Validate KNN Regression across different K values
def validate_knn_regression(x_train, y_train, x_val, y_val, k=range(1, 15)):
    train_mse_list = []
    val_mse_list = []
    time_list = []
    models = []

    for kn in k:
        knn = KNeighborsRegressor(n_neighbors=kn)
        start_time = time.time()
        knn.fit(x_train, y_train)

        models.append(knn)

        # Train and validation predictions
        y_train_pred = knn.predict(x_train)
        y_val_pred = knn.predict(x_val)

        # Calculate MSE for training and validation sets
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        elapsed_time = time.time() - start_time

        train_mse_list.append(train_mse)
        val_mse_list.append(val_mse)
        time_list.append(elapsed_time)

        print("------")
        print(f"For K: {kn}")
        print(f"Training MSE: {train_mse}")
        print(f"Validation MSE: {val_mse}")

        # Plot predictions vs actual values for validation set
        plot_y_yhat(y_val, y_val_pred, 'Validation Predicted vs Expected for K ' + str(kn))

    # Plot MSE vs K values
    plot_k_precision(train_mse_list, val_mse_list)

    # Plot time taken vs K values
    plot_k_time(time_list)

    # Return the model with the lowest validation MSE
    print("Done with model for K " + str(val_mse_list.index(min(val_mse_list)) + 1))
    return models[val_mse_list.index(min(val_mse_list))]


if __name__ == "__main__":
    # Load, process, and split the dataset
    train_data = pd.read_csv('train_data_clean.csv')
    val_data = pd.read_csv('val_data_clean.csv')

    # Prepare the inputs for training and validation (dropping unnecessary columns)
    x_train = prepare_supervised_x_dataset(train_data)
    y_train = prepare_supervised_y_dataset(train_data)

    x_val = prepare_supervised_x_dataset(val_data)
    y_val = prepare_supervised_y_dataset(val_data)

    # Perform feature engineering (add time and distance features)
    x_train = add_time_features(x_train)
    x_train = add_distance_features(x_train)

    x_val = add_time_features(x_val)
    x_val = add_distance_features(x_val)

    # Validate KNN regression models with different K values
    model = validate_knn_regression(x_train, y_train, x_val, y_val, k=range(1, 15))

    # Load test data and prepare it for prediction
    test_data = pd.read_csv("X_test.csv")
    clean_test_data = test_data.drop(columns=['Id'])

    # Change columns to follow the format
    clean_test_data.columns = ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

    # Perform feature engineering on test data
    clean_test_data = add_time_features(clean_test_data)
    clean_test_data = add_distance_features(clean_test_data)

    # Predict on the test dataset using the best model
    y_test_pred = model.predict(clean_test_data)

    # Convert the predictions to a DataFrame and store it as a CSV
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_test_pred_df.insert(0, 'Id', y_test_pred_df.index)
    y_test_pred_df.to_csv('knn_submission.csv', index=False)
    print("Test predictions saved to knn_submission.csv")