import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import random
from typing import Callable
from tqdm import tqdm
#
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

predict_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize((28, 28)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])

class SudokuDigitCNN(nn.Module):
    def __init__(self):
        super(SudokuDigitCNN, self).__init__()
        # Conv Layer 1: More filters for better feature extraction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Conv Layer 2: Deeper feature extraction
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Conv Layer 3: Additional depth
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=9)

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28] (assumed)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = F.relu(self.bn3(self.conv3(x)))             # 7x7 -> 7x7 (no pooling)
        x = torch.flatten(x, 1)                         # Flatten: [batch_size, 64*7*7]
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)                                 # Logits for 9 classes
        return x


def show_random_samples(dataset: Dataset):
    fig = plt.figure(figsize=(10, 1))
    for i in range(10):
        rand_idx = random.randrange(1, len(dataset))
        image, label = dataset[rand_idx]
        ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(image.squeeze(), cmap='gray')
        ax.text(1, 5, dataset.classes[label])
    plt.show()

def train(dataloader: DataLoader, model: nn.Module, loss_fn: Callable, optimizer: optim.Optimizer, num_epochs: int) -> list[float]:
    epoch_loss = []
    model = model.to(device)
    model.train()
    for epoch in tqdm(range(num_epochs), desc='Training'):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss.append(running_loss/len(labels))
    return epoch_loss

def validate(dataloader: DataLoader, model: nn.Module) -> float:
    correct = total = 0
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc='Validate'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    return correct / total

def plot_train_loss(train_loss: list[float]):
    epochs = range(1, len(train_loss)+1)
    plt.plot(epochs, train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.show()

def save_model(model: nn.Module, file_path: str, accuracy: float):
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d%H%M%S")
    accuracy = str(round(accuracy, 4)).replace('0.', '')
    model_name = f"digits_{date_time_str}_{accuracy}.pth"
    print(f"Save trained model: {model_name}")
    torch.save(model.state_dict(), f=f"{file_path}/{model_name}")

def load_model(file_path: str) -> nn.Module:
    model = SudokuDigitCNN()
    model.load_state_dict(torch.load(file_path, weights_only=True))
    return model

def predict(image: np.ndarray, model: nn.Module) -> tuple[int,float]:
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    image_pil = Image.fromarray(image)
    model.eval()
    with torch.inference_mode():
        image_transformed = predict_transform(image_pil)
        pred = model(image_transformed.unsqueeze(dim=1))
        label = torch.argmax(pred, dim=1).item()
        predicted_digit = classes[label]
        probability = torch.softmax(pred, dim=1)[0][label]
    return predicted_digit, probability

if __name__ == '__main__':
    print("Model: digits.py for training PyTorch model to recognize digits.")
    print("Please run solver.py")