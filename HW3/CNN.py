import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

# (TODO) Design your CNN, it can only be less than 3 convolution layers
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Output: (16, 224, 224)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # Output: (16, 112, 112)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Output: (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # Output: (32, 56, 56)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # Output: (64, 28, 28)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                      # Output: 64*28*28 = 50176
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x
    
        # raise NotImplementedError

# (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            
        outputs = model(images)          
        loss = criterion(outputs, labels)
        loss.backward()                  
        optimizer.step()                 

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / total_batches
    return avg_loss


# (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(val_loader)

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / total_batches
    accuracy = correct / total
    return avg_loss, accuracy
    

# (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
def test(model: CNN, test_loader: DataLoader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(zip(image_ids, predicted.cpu().numpy()))
    predictions.sort(key=lambda x: int(x[0]))
    with open('CNN.csv', 'w') as f:
        f.write('id,prediction\n')
        for image_id, pred in predictions:
            f.write(f'{image_id},{pred}\n')
    print(f"Predictions saved to 'CNN.csv'")
    return