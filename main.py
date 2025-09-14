import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time

if __name__ == "__main__":
    DATA_DIR = "F:/ai_challenge/data"
    MODEL_PATH = "F:/ai_challenge/models/cnn_animals.pth"
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    class DeepCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2,2)
            self.fc1 = nn.Linear(128*16*16, 256)
            self.fc2 = nn.Linear(256, num_classes)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = DeepCNN(len(train_data.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        print(f"Começando época {epoch+1}")
        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_time = time.time() - start_time
        print(f"Época {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Acurácia: {correct/total:.4f} - Tempo total: {epoch_time:.2f}s")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"acurácia: {correct/total:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")
