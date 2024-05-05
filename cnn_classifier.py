import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader


class ConvolutionalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def train(model, device, train_data, nb_epochs, batch_size, learning_rate):
    print(f"Training on device: {device}")

    loss_values = []
    epoch_values = []

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(nb_epochs):
        running_loss = 0.
        for batch in data_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(train_data)
        epoch_values.append(epoch)
        loss_values.append(epoch_loss)
        
        print(f'[epoch {epoch}] epoch loss = {epoch_loss:.4f}', end='\r')

    return epoch_values, loss_values
