import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from model import prepare_supervised_x_dataset, prepare_supervised_y_dataset, process_and_store_splits, add_time_features, add_distance_features

train_mse_list = []
val_mse_list = []
models = []


# Plotting the RMSE against polynomial degrees
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
    plt.ylabel('Root Mean Square Error (RMSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Polynomial regression validation across multiple degrees
def validate_poly_regression(x_train, y_train, x_val, y_val, regressor=None, degrees=range(5, 15), max_features=None):
    best_rmse = float('inf')  # Start with infinity as the worst RMSE
    best_model = None
    best_degree = None

    # Loop through each degree of polynomial and validate
    for degree in degrees:
        print(f"Validating polynomial degree: {degree}")
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        # Create a pipeline with feature engineering, polynomial features, scaling, and the regressor (Ridge)
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardization
            ('poly_features', poly),  # Polynomial transformation
            ('regressor', regressor)  # Ridge regression
        ])

        # Fit the pipeline on the training data
        model_pipeline.fit(x_train, y_train)

        # Predict on both training and validation sets
        y_train_pred = model_pipeline.predict(x_train)
        y_val_pred = model_pipeline.predict(x_val)

        # Calculate the RMSE for both training and validation
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        train_mse_list.append(train_rmse)
        val_mse_list.append(val_rmse)

        # Print degree, feature count, and performance metrics
        num_features = model_pipeline.named_steps['poly_features'].n_output_features_
        print(f"Degree: {degree}, Features: {num_features}, Train RMSE: {train_rmse}, Val RMSE: {val_rmse}")

        # Keep track of the best performing model based on validation RMSE
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = model_pipeline
            best_degree = degree

    print(f"Best polynomial degree: {best_degree} with RMSE: {best_rmse}")

    return best_model, best_rmse


if __name__ == "__main__":
    # Load, process, and split the dataset
    process_and_store_splits('X_train.csv', 0.05, 0.05, 0.9)

    train_data = pd.read_csv('train_data_clean_poli.csv')
    val_data = pd.read_csv('val_data_clean_poli.csv')

    # Prepare the inputs for training and validation datasets
    x_train = prepare_supervised_x_dataset(train_data)
    y_train = prepare_supervised_y_dataset(train_data)

    x_val = prepare_supervised_x_dataset(val_data)
    y_val = prepare_supervised_y_dataset(val_data)

    # Perform feature engineering (add time and distance features)
    x_train = add_time_features(x_train)
    x_train = add_distance_features(x_train)
    
    x_val = add_time_features(x_val)
    x_val = add_distance_features(x_val)

    # Validate polynomial regression models with degrees from 1 to 10
    regressor = RidgeCV(alphas=(0.1, 1.0, 10.0))  # Use Ridge regression with cross-validation for alpha
    best_model, best_rmse = validate_poly_regression(x_train, y_train, x_val, y_val, regressor=regressor, degrees=range(1, 11))

    # Plot RMSE vs. polynomial degrees
    plot_n_degrees()

    # Process and predict test data using the best model
    test_data = pd.read_csv("X_test.csv")
    clean_test_data = test_data.drop(columns=['Id'])
    clean_test_data.columns = ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

    # Perform feature engineering on the test data
    clean_test_data = add_time_features(clean_test_data)
    clean_test_data = add_distance_features(clean_test_data)

    # Predict on the test dataset using the best polynomial model
    y_test_pred = best_model.predict(clean_test_data)

    # Save the predictions to a CSV file
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_test_pred_df.insert(0, 'Id', y_test_pred_df.index)
    y_test_pred_df.to_csv('polynomial_submission.csv', index=False)
    print("Test predictions saved to polynomial_submission.csv")
