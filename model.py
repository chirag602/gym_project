import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Define the model
class FitnessModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FitnessModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Step 2: Load preprocessed data
def load_data():
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, label_encoder

# Step 3: Train the model
def train_model(X_train, y_train, model, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        inputs, labels = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Step 4: Evaluate the model
def evaluate_model(X_test, y_test, model, device):
    model.eval()
    inputs, labels = X_test.to(device), y_test.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == labels).float().mean().item()
    return accuracy

# Step 5: Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Step 6: Load the model
def load_model(model_path, input_size, num_classes):
    model = FitnessModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model