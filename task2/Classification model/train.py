import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import models
import numpy as np

#constants
DATA_DIR = "animal10/Animals-10"
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 15
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load dataset
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

#function for balance dataset by selecting the same number of samples for each class
def balance_dataset(dataset):
    class_counts = np.bincount([label for _, label in dataset.samples])
    min_class_count = min(class_counts)  #find the minimum count
    print(min_class_count)
    balanced_indices = []

    for class_idx in range(len(class_counts)):
        class_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
        balanced_indices += np.random.choice(class_indices, min_class_count, replace=False).tolist()
    
    return torch.utils.data.Subset(dataset, balanced_indices)

balanced_dataset = balance_dataset(full_dataset)

#data loaders
train_size = int(0.8 * len(balanced_dataset))
val_size = len(balanced_dataset) - train_size
train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#accuracy function
def compute_accuracy(loader, model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

#training loop
best_val_acc = 0.0
train_losses, val_accuracies = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    loop = tqdm(train_loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    val_acc = compute_accuracy(val_loader, model)
    train_losses.append(train_loss)
    val_accuracies.append(val_acc)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2%}, Val Accuracy: {val_acc:.2%}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

#plot results
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.title("Training Progress")
plt.show()
