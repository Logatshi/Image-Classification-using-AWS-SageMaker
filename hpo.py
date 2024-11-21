# TODO: Import your dependencies.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import os
from torchvision import datasets
from tqdm import tqdm


def test(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    model.to(device)
    test_loss = correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2%}")


def train(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Train the model using the training data loader.
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print(f"Starting epoch {epoch + 1}/{epochs}")

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 100 == 0:  # Log every 100 batches
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")


def net(num_classes=133):
    """
    Initialize and return a pre-trained ResNet50 model, adapted for the specified number of classes.
    """
    model = models.resnet50(pretrained=True)
    # Freeze model weights except for the last fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_data_loaders(data_dir, batch_size, image_size=224):
    """
    Create and return data loaders for training, validation, and testing datasets.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    valid_data = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Initialize a model by calling the net function
    model = net(args.num_classes)

    # Create loss criterion and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.learning_rate)

    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size, args.image_size)

    # Train the model
    train(model, train_loader, loss_criterion, optimizer, device, epochs=args.epochs)

    # Test the model
    test(model, test_loader, loss_criterion, device)

    # Save the trained model
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_path, "model.pth"))
    print("Model weights saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help='Training batch size')
    parser.add_argument("--num_classes", type=int, default=133, help='Number of output classes')
    parser.add_argument("--epochs", type=int, default=10, help='Number of epochs')
    parser.add_argument("--device", type=str, default="cuda", help='Device to use for training (cuda or cpu)')
    parser.add_argument("--learning_rate", type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument("--model_path", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"), help='Path to save the model')
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "./data"), help='Path to dataset')
    parser.add_argument("--image_size", type=int, default=224, help='Image size for resizing input images')

    args = parser.parse_args()
    main(args)