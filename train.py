import torch
from model import FitnessModel, train_model, save_model
import torch.nn as nn
import torch.optim as optim

# Initialize model with new architecture
model = FitnessModel(input_size=99, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Load your training data
X_train = torch.load('X_train.pt')  # or however you load your data
y_train = torch.load('y_train.pt')

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_model(X_train, y_train, model, criterion, optimizer, device)

# Save the new model
save_model(model, 'fitness_model.pth') 