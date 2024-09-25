import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Fonction pour encoder en one-hot
def one_hot_encode(number):
    """Converts a number (1-9) into a one-hot vector of size 9."""
    one_hot = np.zeros(9, dtype=int)
    if number > 0:
        one_hot[number - 1] = 1
    return one_hot

# Fonction pour préparer les grilles
def prepare_9_channel_grid(data_row):
    """Transforms a string of 81 digits into a 9x9x9 tensor suitable for CNN processing."""
    grid = np.array(list(map(int, data_row))).reshape(9, 9)
    channels = np.zeros((9, 9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            channels[:, i, j] = one_hot_encode(grid[i, j])
    return channels

# Définition du modèle de réseau de neurones
class SudokuCNN(nn.Module):
    def __init__(self):
        super(SudokuCNN, self).__init__()
        # self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding="same")
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        # self.conv4 = nn.Conv2d(256, 9, kernel_size=3, padding="same")  # Output 9 channels for the 9 numbers

        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(64, 9, kernel_size=3, padding="same")  # Output 9 channels for the 9 numbers

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return F.log_softmax(x, dim=1)  # Apply log_softmax across the channels representing the numbers

# Chargement des données Sudoku
puzzles_path = 'sudoku_puzzles.csv'
solutions_path = 'sudoku_solutions.csv'

# Chargement des données
puzzles_9x9 = pd.read_csv(puzzles_path, header=None)
solutions_9x9 = pd.read_csv(solutions_path, header=None)

# Préparation des données
X = np.stack([prepare_9_channel_grid(row[0]) for row in puzzles_9x9.values])
print("Shape of XXXXX:")
print(X[0])
Y = np.stack([prepare_9_channel_grid(row[0]) for row in solutions_9x9.values])
print("Shape of YYYYYYY:")
print(Y[0])
Y = np.argmax(Y, axis=1)  # Convert one-hot to class numbers for each cell
print("Shape of indice:")
print(Y[0])

# Convertir en tenseurs
X_train = torch.tensor(X[:80000], dtype=torch.float32)
Y_train = torch.tensor(Y[:80000], dtype=torch.long)
X_val = torch.tensor(X[80000:], dtype=torch.float32)
Y_val = torch.tensor(Y[80000:], dtype=torch.long)
print("Nombre d'exemples d'entraînement:", len(X_train))
print("Nombre d'exemples de validation:", len(X_val))
print(X_train[0])
print(X_train.size())
#
print(Y_train[0])
#
print(Y_train.size())

# Créer des DataLoader
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
batch_size = 5000
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialisation du modèle, de la fonction de perte et de l'optimiseur
model = SudokuCNN()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Fonction pour entraîner et évaluer le modèle
def train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, epochs=3000):
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        total_train_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, 9), Y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs.view(-1, 9), Y_batch.view(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch % 50 == 0:
            print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses

# Lancer l'entraînement et l'évaluation
train_losses, val_losses = train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, epochs=3000)

# Tracer la perte d'entraînement
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Loss Entraînement')
plt.plot(val_losses, label='Loss Validation')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.title("Perte d'entraînement et de validation")
plt.legend()
plt.show()


def evaluate_model(model, val_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch)
            # Assurez-vous que les dimensions sont les mêmes pour les prédictions et les étiquettes
            _, predicted = torch.max(outputs, 1)  # Utilisez la bonne dimension pour torch.max
            total_correct += (predicted.view(-1) == Y_batch.view(-1)).sum().item()
            total_samples += Y_batch.numel()  # Utilisez numel() pour obtenir le nombre total d'éléments dans le batch
    accuracy = total_correct / total_samples
    print(f'Accuracy: {accuracy:.4f}')

evaluate_model(model, val_loader)
