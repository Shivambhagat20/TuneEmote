import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Define the EmotionModel class (similar to what you would have in model.py)
class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)  # Adjust the size based on your image size
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the necessary transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize images to match the input size of the model
    transforms.Grayscale(),       # Convert images to grayscale
    transforms.ToTensor(),        # Convert images to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image data
])

# Set paths for the dataset
train_dir = 'FER2013/train'
test_dir = 'FER2013/test'

# Create datasets and dataloaders
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model, loss function, and optimizer
model = EmotionModel(num_classes=7)  # 7 emotions in the FER-2013 dataset
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Update weights
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print epoch statistics
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Save the trained model after training is complete
torch.save(model.state_dict(), 'emotion_model.pth')
print("Model saved as 'emotion_model.pth'")

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
