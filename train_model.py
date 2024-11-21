import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import ImageFile

# To prevent potential crashes on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_model(batch_size, epochs, lr, data_path, model_dir, model_name="resnet18"):
    # Check if the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} not found")

    # Data transformations (augmentation, normalization)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validate dataset and structure
    try:
        train_dataset = datasets.ImageFolder(root=data_path, transform=transform)
        if len(train_dataset.classes) == 0:
            raise ValueError("No class subdirectories found in the data path.")
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {data_path}. Error: {e}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Define the model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Adjust output layer size
    else:
        raise ValueError("Unsupported model name")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            try:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            except Exception as e:
                print(f"Error during training batch: {e}")

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

    # Save the model to the specified directory
    model_save_path = os.path.join(model_dir, "model.pth")
    os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add SageMaker specific arguments with names matching SageMaker hyperparameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-path', type=str, default='/opt/ml/input/data/training', help='Path to training data')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model', help='Directory to save the trained model')
    parser.add_argument('--model-name', type=str, default='resnet18', help='Model architecture to use (default: resnet18)')

    args = parser.parse_args()

    # Call the training function with parsed arguments
    train_model(args.batch_size, args.epochs, args.lr, args.data_path, args.model_dir, args.model_name)