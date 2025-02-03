import torch
import torch.nn as nn
import torch.optim as optim
from model import (
    FitnessModel,
    load_data,
    train_model,
    evaluate_model,
    save_model
)

if __name__ == '__main__':
    # Load preprocessed data
    X_train, y_train, X_test, y_test, label_encoder = load_data()

    # Get input size and number of classes
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FitnessModel(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    train_model(X_train, y_train, model, criterion, optimizer, device, num_epochs=10)

    # Evaluate the model
    print("Evaluating the model...")
    accuracy = evaluate_model(X_test, y_test, model, device)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    print("Saving the model...")
    save_model(model, "fitness_model.pth")
    print("Model saved successfully!")