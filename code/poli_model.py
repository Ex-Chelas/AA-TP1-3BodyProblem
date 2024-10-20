import itertools

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from model import prepare_supervised_x_dataset, prepare_supervised_y_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the PyTorch model
class PolyRegressionModel(nn.Module):
    def __init__(self, input_size, output_size=6):
        super(PolyRegressionModel, self).__init__()
        # Linear layer to map the expanded polynomial features to the output
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


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


def polynomial_features(x, degree):
    """
    Manually generate polynomial features for tensor x, up to the specified degree.
    This implementation includes interaction terms.

    Parameters:
    - x: A PyTorch tensor of shape (n_samples, n_features)
    - degree: The maximum degree of the polynomial features

    Returns:
    - A new tensor with the polynomial features of shape (n_samples, n_polynomial_features)
    """
    poly_features = [x]  # Start with the original features

    n_features = x.shape[1]

    # Generate polynomial terms for each degree from 2 to the specified degree
    for deg in range(2, degree + 1):
        # Generate combinations of features for the current degree
        for comb in itertools.combinations_with_replacement(range(n_features), deg):
            new_feature = torch.ones(x.size(0), device=x.device)
            for idx in comb:
                new_feature *= x[:, idx]
            poly_features.append(new_feature.unsqueeze(1))

    # Concatenate the polynomial features along the feature axis
    return torch.cat(poly_features, dim=1)


def validate_poly_regression(x_train, y_train, x_val, y_val, degrees=range(1, 10), max_features=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_rmse = float('inf')
    best_model = None
    best_degree = None

    for degree in degrees:
        print(f"Evaluating degree {degree}...")

        # Convert the Pandas DataFrame to NumPy array before creating tensors
        x_train_np = x_train.to_numpy()
        x_val_np = x_val.to_numpy()

        # Generate polynomial features manually using the custom function
        x_train_poly = polynomial_features(torch.tensor(x_train_np, dtype=torch.float32).to(device), degree)
        x_val_poly = polynomial_features(torch.tensor(x_val_np, dtype=torch.float32).to(device), degree)

        # Safely handle y_train and y_val to avoid the warning
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.clone().detach().to(device)
        else:
            y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

        if isinstance(y_val, torch.Tensor):
            y_val = y_val.clone().detach().to(device)
        else:
            y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

        # Create PyTorch model
        model = PolyRegressionModel(input_size=x_train_poly.shape[1], output_size=6).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        model.train()
        for epoch in range(100):  # You can adjust the number of epochs
            optimizer.zero_grad()
            outputs = model(x_train_poly)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        with torch.no_grad():
            y_train_pred = model(x_train_poly)
            y_val_pred = model(x_val_poly)

        train_rmse = torch.sqrt(criterion(y_train_pred, y_train)).item()
        val_rmse = torch.sqrt(criterion(y_val_pred, y_val)).item()

        train_mse_list.append(train_rmse)
        val_mse_list.append(val_rmse)

        print(f"Degree: {degree}, Train RMSE: {train_rmse}, Validation RMSE: {val_rmse}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = model
            best_degree = degree

    print(f"Best degree: {best_degree} with Validation RMSE: {best_rmse}")
    return best_model, best_rmse


if __name__ == "__main__":
    # Load, process, and split the dataset
    # process_and_store_splits('code\\X_train.csv')
    train_data = pd.read_csv('code\\train_data_clean.csv')
    val_data = pd.read_csv('code\\val_data_clean.csv')

    # Prepare the inputs for training and validation
    x_train = prepare_supervised_x_dataset(train_data)
    y_train = prepare_supervised_y_dataset(train_data)

    x_val = prepare_supervised_x_dataset(val_data)
    y_val = prepare_supervised_y_dataset(val_data)

    # Validate using polynomial regression with PyTorch
    best_model, best_rmse = validate_poly_regression(x_train, y_train, x_val, y_val, degrees=range(1, 10))

    # For testing on GPU, move data to GPU
    test_data = pd.read_csv("code\\X_test.csv").drop(columns=['Id'])
    clean_test_data = torch.tensor(test_data.values, dtype=torch.float32).to(device)

    # Predict on the test data using the best model
    best_model.eval()
    with torch.no_grad():
        y_test_pred = best_model(clean_test_data).cpu().numpy()

    # Save predictions
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    y_test_pred_df.insert(0, 'Id', y_test_pred_df.index)
    y_test_pred_df.to_csv('code\\baseline-model.csv', index=False)
