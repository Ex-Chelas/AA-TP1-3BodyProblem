import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from model import prepare_supervised_x_dataset, prepare_supervised_y_dataset, plot_y_yhat, process_and_store_splits

train_mse_list = []
val_mse_list = []
models = []


def plot_n_degrees():
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_mse_list) + 1), train_mse_list, label="Training RMSE", marker='o')
    plt.plot(range(1, len(val_mse_list) + 1), val_mse_list, label="Validation RMSE", marker='o')

    min_val_mse = min(val_mse_list)
    min_dg = val_mse_list.index(min_val_mse) + 1
    plt.axhline(y=min_val_mse, color='r', linestyle='--',
                label=f'Lowest RMSE = {min_val_mse:.4f} at degree:{min_dg}')

    plt.title('RMSE vs Degrees for Training and Validation Sets')
    plt.xlabel('Degrees')
    plt.ylabel('Root Mean Square Deviation (RMSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


def validate_poly_regression(x_train, y_train, x_val, y_val, regressor=None, degrees=range(5, 15), max_features=None):
    best_rmse = 2.0
    best_model = None
    best_degree = None

    # Loop through each degree and validate the model
    for degree in degrees:
        print(f"{degree} in the tank, {degree} in the tank. Swing it around and it becomes")
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        # Create a pipeline with polynomial features, scaling, and the regressor
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly_features', poly),
            ('regressor', regressor)
        ])

        # Fit the model pipeline on the sampled training data
        model_pipeline.fit(x_train, y_train)

        y_train_pred = model_pipeline.predict(x_train)
        y_val_pred = model_pipeline.predict(x_val)

        # Calculate MSE for training and validation sets
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mse_list.append(train_rmse)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mse_list.append(val_rmse)

        # Print number of polynomial features
        num_features = model_pipeline.named_steps['poly_features'].n_output_features_
        print(f"Degree: {degree}, Features: {num_features}, T_RMSE: {train_rmse}, V_RMSE: {val_rmse}")

        # Check if the number of features exceeds max_features, if specified
        if max_features and num_features > max_features:
            print(f"Skipping degree {degree} due to feature count exceeding {max_features}")
            continue

        # Keep track of the best model and RMSE
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = model_pipeline
            best_degree = degree

    print(f"Best degree: {best_degree} with RMSE: {best_rmse}")

    return best_model, best_rmse


if __name__ == "__main__":
    # Load, process, and split the dataset
    process_and_store_splits('X_train.csv', 0.1, 0.1, 0.8)

    train_data = pd.read_csv('train_data_clean.csv')
    val_data = pd.read_csv('val_data_clean.csv')

    # Prepare the inputs for training and predictions (dropping unnecessary columns)
    x_train = prepare_supervised_x_dataset(train_data)
    y_train = prepare_supervised_y_dataset(train_data)

    x_val = prepare_supervised_x_dataset(val_data)
    y_val = prepare_supervised_y_dataset(val_data)

    #test_data = pd.read_csv('test_data_clean.csv')  # TODO: ONLY USE THIS ON FINAL VERSION, shall remain untouched
    #x_test = prepare_supervised_x_dataset(test_data)
    #y_test = prepare_supervised_y_dataset(test_data)

    # Load test data and prepare it for prediction
    #plot_k_precision()
    #plot_k_time()

    res = validate_poly_regression(x_train, y_train, x_val, y_val, regressor=RidgeCV(), degrees=range(1, 7))
    plot_n_degrees()
    test_data = pd.read_csv("X_testa.csv")
    clean_test_data = test_data.drop(columns=['Id'])

    # Change columns to follow the format
    clean_test_data.columns = ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

    # Predict on the test data
    print("Done with model for K " + str(val_mse_list.index(min(val_mse_list)) + 1))
    model = models[val_mse_list.index(min(val_mse_list)) + 1]
    y_test_pred = model.predict(clean_test_data)

    # Convert the predictions to a DataFrame and store it as a CSV
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_test_pred_df.insert(0, 'Id', y_test_pred_df.index)

    y_test_pred_df.to_csv('baseline-model.csv', index=False)
