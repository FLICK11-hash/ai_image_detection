import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def limit_dataset(dataset, limit):
    if limit is None:
        return dataset

    real_indices = []
    fake_indices = []

    for i, (_, label) in enumerate(dataset.samples):
        if label == 0:
            real_indices.append(i)
        else:
            fake_indices.append(i)

    half = limit // 2
    selected = real_indices[:half] + fake_indices[:half]

    return Subset(dataset, selected)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


def main():
    train_dir = "data/train"
    test_dir = "data/test"
    limit = 10000
    epochs = 10
    batch_size = 64

    Path("results").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_data = limit_dataset(train_data, limit)
    test_data = limit_dataset(test_data, 2000)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        metrics = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {metrics['accuracy']:.4f}")

    final_metrics = evaluate(model, test_loader, device)
    final_metrics["model"] = "Simple CNN"
    final_metrics["runtime_seconds"] = time.time() - start

    pd.DataFrame([final_metrics]).to_csv("results/cnn_results.csv", index=False)
    torch.save(model.state_dict(), "results/simple_cnn_model.pth")

    print(pd.DataFrame([final_metrics]))


if __name__ == "__main__":
    main()