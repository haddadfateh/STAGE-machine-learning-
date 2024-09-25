import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def one_hot_encode(number):
    """Convert a number to one-hot encoding over 9 bits for digits 0-9, where 0 is represented as all zeros."""
    one_hot = np.zeros(9, dtype=int)
    if number > 0:
        one_hot[number - 1] = 1
    return one_hot

def prepare_one_hot_vector_9x9(data_string):
    """Convert a string representation of a Sudoku row into a one-hot encoded 81*9 vector."""
    grid = np.array([int(num) for num in data_string]).reshape(81)
    one_hot_grid = np.array([one_hot_encode(number) for number in grid])
    return one_hot_grid.flatten()

def decode_sudoku_grid(one_hot_encoded):
    """Translate a one-hot encoded Sudoku grid back to its numerical representation with zeros intact."""
    grid = []
    for i in range(0, len(one_hot_encoded), 9):
        block = one_hot_encoded[i:i+9]
        if np.sum(block) == 0:
            grid.append(0)
        else:
            grid.append(np.argmax(block) + 1)
    return np.array(grid).reshape(9, 9)

# Load the puzzles and solutions
puzzles_9x9 = pd.read_csv('sudoku_puzzles.csv', dtype=str)
solutions_9x9 = pd.read_csv('sudoku_solutions.csv', dtype=str)

# Prepare X and Y
X = np.array([prepare_one_hot_vector_9x9(row[0]) for row in puzzles_9x9.to_numpy()])
Y = np.array([prepare_one_hot_vector_9x9(row[0]) for row in solutions_9x9.to_numpy()])
Y = np.array([row.reshape(-1, 9).argmax(axis=1) for row in Y])  # Convert one-hot to indices

# Split the data
train_size = 90000
X_train = torch.tensor(X[:train_size], dtype=torch.float32)
Y_train = torch.tensor(Y[:train_size], dtype=torch.long)
X_val = torch.tensor(X[train_size:], dtype=torch.float32)
Y_val = torch.tensor(Y[train_size:], dtype=torch.long)

# Create DataLoader for training and validation sets
batch_size = 10000
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class SudokuNN(nn.Module):
    def __init__(self):
        super(SudokuNN, self).__init__()
        self.fc1 = nn.Linear(729, 1458)
        self.fc2 = nn.Linear(1458, 2916)
        self.fc3 = nn.Linear(2916, 2916)
        self.fc4 = nn.Linear(2916, 729)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x).view(-1, 81, 9)
        return F.softmax(x, dim=2)

model = SudokuNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=1000):
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, 9), Y_batch.view(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                val_outputs = model(X_batch)
                val_loss = criterion(val_outputs.view(-1, 9), Y_batch.view(-1))
                val_losses.append(val_loss.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch + 1}: Train Loss: {np.mean(train_losses)}, Val Loss: {np.mean(val_losses)}')

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return train_losses, val_losses

train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=1000)

def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted_classes = outputs.max(2)
            correct += (predicted_classes.view(-1) == Y_batch.view(-1)).sum().item()
            total += Y_batch.size(0) * Y_batch.size(1)
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

evaluate_model(model, val_loader)
