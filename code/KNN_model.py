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

from model import prepare_supervised_x_dataset, prepare_supervised_y_dataset, plot_y_yhat, process_and_store_splits

train_mse_list = []
val_mse_list = []
time_list = []
models = []


def plot_k_precision():
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


def plot_k_time():
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16), time_list, label="Time Taken", marker='o', color='r')
    plt.title('Time Taken vs K Value for Training and Predicting')
    plt.xlabel('K Value')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()


def knn_regression(X_train, Y_train, k=15):  #range(1, 15)
    knn = KNeighborsRegressor(n_neighbors=k)

    knn.fit(X_train, Y_train)

    models.append(knn)
    return knn


if __name__ == "__main__":
    # Load, process, and split the dataset
    #process_and_store_splits('X_train.csv')

    train_data = pd.read_csv('train_data_clean.csv')
    val_data = pd.read_csv('val_data_clean.csv')
    test_data = pd.read_csv('test_data_clean.csv')  # TODO: ONLY USE THIS ON FINAL VERSION, shall remain untouched

    # Prepare the inputs for training and predictions (dropping unnecessary columns)
    x_train = prepare_supervised_x_dataset(train_data)
    y_train = prepare_supervised_y_dataset(train_data)

    x_val = prepare_supervised_x_dataset(val_data)
    y_val = prepare_supervised_y_dataset(val_data)

    x_test = prepare_supervised_x_dataset(test_data)
    y_test = prepare_supervised_y_dataset(test_data)
    for i in range(1, 16):
        # Build and train the model
        start_time = time.time()
        model = knn_regression(x_train, y_train, i)

        # Train and validation predictions
        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val)
        y_test_pred = model.predict(x_test)

        # Calculate MSE for training and validation sets
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        end_time = time.time()

        elapsed_time = time.time() - start_time

        train_mse_list.append(train_mse)
        val_mse_list.append(val_mse)
        time_list.append(elapsed_time)

        print("------")
        print(f"For K: {i}")
        print(f"Training MSE: {train_mse}")
        print(f"Validation MSE: {val_mse}")
        print(f"Teste MSE: {test_mse}")

        # plot_y_yhat(y_train, y_train_pred, 'Train Predicted vs Expected')
        plot_y_yhat(y_val, y_val_pred, 'Validation Predicted vs Expected for K ' + str(i))

    # Load test data and prepare it for prediction
    plot_k_precision()
    plot_k_time()
    test_data = pd.read_csv("X_testa.csv")
    clean_test_data = test_data.drop(columns=['Id'])

    # Change columns to follow the format
    clean_test_data.columns = ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

    # Predict on the test data
    print("Done with model for K "+str(val_mse_list.index(min(val_mse_list)) + 1))
    model = models[val_mse_list.index(min(val_mse_list)) + 1]
    y_test_pred = model.predict(clean_test_data)

    # Convert the predictions to a DataFrame and store it as a CSV
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_test_pred_df.insert(0, 'Id', y_test_pred_df.index)

    y_test_pred_df.to_csv('baseline-model.csv', index=False)
