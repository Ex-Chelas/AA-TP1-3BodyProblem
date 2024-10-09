import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Load the dataset
def load_data_frame(path):
    return pd.read_csv(path)

df_train = load_data_frame('../data/X_train.csv')
df_test = load_data_frame('../data/X_test.csv')

# Add velocity columns to the test data with values initialized to zero (as described)
df_test['v_x_1'] = 0
df_test['v_y_1'] = 0
df_test['v_x_2'] = 0
df_test['v_y_2'] = 0
df_test['v_x_3'] = 0
df_test['v_y_3'] = 0

# Prepare the test set for predictions: make sure it matches the feature columns of the train set
df_test_prepared = df_test.rename(columns={
    'x0_1': 'x_1', 'y0_1': 'y_1', 'x0_2': 'x_2', 'y0_2': 'y_2', 'x0_3': 'x_3', 'y0_3': 'y_3'
})

# Feature columns in the dataset
feature_columns = ['t', 'x_1', 'y_1', 'v_x_1', 'v_y_1', 'x_2', 'y_2', 'v_x_2', 'v_y_2', 'x_3', 'y_3', 'v_x_3', 'v_y_3']

# Target columns in the datasets
target_columns = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

# Train the model on the entire train data
def train_and_predict(df_train, df_test):
    # Extract features and targets
    X_train = df_train[feature_columns] # Use all features for training
    y_train = df_train[target_columns]  # Predict future positions

    X_test = df_test_prepared[feature_columns]  # Use the same structure as train for test predictions
    
    # Create a pipeline for scaling and regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('regressor', LinearRegression())
    ])
    
    # Train the model on the entire train set
    pipeline.fit(X_train, y_train)
    
    # Predict positions for the test set
    y_pred_test = pipeline.predict(X_test)
    
    return y_pred_test

# Predict future positions
test_predictions = train_and_predict(df_train, df_test_prepared)

# Convert predictions to DataFrame and assign proper column names
df_test_predictions = pd.DataFrame(test_predictions, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])

# Add the Id column to the predictions DataFrame
df_test_predictions['Id'] = df_test['Id']

# Save the predictions to CSV
df_test_predictions.to_csv('../data/predicted_positions.csv', index=False)

print("Test predictions saved.")
