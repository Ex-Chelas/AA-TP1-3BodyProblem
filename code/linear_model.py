import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from model import prepare_supervised_x_dataset, prepare_supervised_y_dataset, plot_y_yhat, process_and_store_splits, add_time_features, add_distance_features, add_squared_distance_features, add_inverse_distance_features, add_ratio_of_distance_features

# Function to build and train the linear regression model
def build_and_train_model(x_train, y_train):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardization
        ('regressor', LinearRegression())  # Linear regression model
    ])
    model_pipeline.fit(x_train, y_train)
    return model_pipeline

# Function to evaluate the model with added features
def evaluate_features(df, y_train, features_to_add, x_val, y_val):
    # Original dataset without additional features
    base_df = df.copy()
    
    # Initialize variables to track the best model
    best_mse = float('inf')
    best_features = []
    
    # Define the feature functions
    feature_functions = {
        'time_features': add_time_features,
        'distance_features': add_distance_features,
        'squared_distance_features': add_squared_distance_features,
        'inverse_distance_features': add_inverse_distance_features,
        'ratio_of_distance_features': add_ratio_of_distance_features
    }

    for feature_name in features_to_add:
        # Add the feature to the dataset
        df = feature_functions[feature_name](df)
        
        # Train the model
        model = build_and_train_model(x_train, y_train)
        
        # Validate the model
        y_val_pred = model.predict(x_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        
        print(f"Feature added: {feature_name}, Validation MSE: {val_mse}")
        
        # If the new feature improves the performance, update the best model and features
        if val_mse < best_mse:
            best_mse = val_mse
            best_features.append(feature_name)
            print(f"New best feature set: {best_features}, Best Validation MSE: {best_mse}")
    
    return best_features, best_mse


if __name__ == "__main__":
    # Load, process, and split the dataset
    process_and_store_splits('X_train.csv', 0.4, 0.3, 0.3)
    train_data = pd.read_csv('train_data_clean.csv')
    val_data = pd.read_csv('val_data_clean.csv')

    # Prepare the inputs for training and predictions (dropping unnecessary columns)
    x_train = prepare_supervised_x_dataset(train_data)
    y_train = prepare_supervised_y_dataset(train_data)

    x_val = prepare_supervised_x_dataset(val_data)
    y_val = prepare_supervised_y_dataset(val_data)

    # Add features to the training and validation datasets
    x_train = add_time_features(x_train)
    x_val = add_time_features(x_val)

    # Evaluate features and get the best set of features
    best_features, best_mse = evaluate_features(x_train, y_train, ['time_features'], x_val, y_val)
    print(f"Best features: {best_features}, Best Validation MSE: {best_mse}")

    # Build and train the model with the best features
    model = build_and_train_model(x_train, y_train)

    # Train and validation predictions
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

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

    # Add features to the test dataset
    clean_test_data = add_time_features(clean_test_data)

    # Predict on the test data
    y_test_pred = model.predict(clean_test_data)

    # Convert the predictions to a DataFrame and store it as a CSV
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_test_pred_df.insert(0, 'Id', y_test_pred_df.index)

    y_test_pred_df.to_csv('baseline-model.csv', index=False)
    print("Predictions saved to baseline-model.csv")
