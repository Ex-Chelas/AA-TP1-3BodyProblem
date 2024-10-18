import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from model import prepare_supervised_x_dataset, prepare_supervised_y_dataset, plot_y_yhat


# Function to build and train the linear regression model
def build_and_train_model(x_train, y_train):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardization
        ('regressor', LinearRegression())  # Linear regression model
    ])

    model_pipeline.fit(x_train, y_train)
    return model_pipeline


if __name__ == "__main__":
    # Load, process, and split the dataset
    # process_and_store_splits('X_train.csv')

    train_data = pd.read_csv('train_data_clean.csv')
    val_data = pd.read_csv('val_data_clean.csv')
    # test_data = pd.read_csv('test_data_clean.csv') TODO: ONLY USE THIS ON FINAL VERSION, shall remain untouched

    # Prepare the inputs for training and predictions (dropping unnecessary columns)
    x_train = prepare_supervised_x_dataset(train_data)
    y_train = prepare_supervised_y_dataset(train_data)

    x_val = prepare_supervised_x_dataset(val_data)
    y_val = prepare_supervised_y_dataset(val_data)

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
