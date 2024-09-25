

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Fonction pour convertir un chiffre en vecteur one-hot 3-bits
# Les chiffres possibles sont 0, 1, 2, et 3, où 0 peut représenté une cas vide.
def one_hot_encode(number):

    mapping = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [0, 1, 0],
        3: [0, 0, 1]
    }
    return mapping.get(number, [0, 0, 0])

# Fonction pour charger les données depuis des fichiers CSV et les préparer
#Cette fonction charge les grilles de Sudoku complètes et incomplètes à partir de fichiers CSV, les encode en format one-hot, et les prépare pour l'entraînement
def load_and_prepare_data(complete_path, incomplete_path):

    complete_df = pd.read_csv(complete_path)
    incomplete_df = pd.read_csv(incomplete_path)
# Initialisation des Listes pour les Entrées et Sorties
    X, Y = [], []
# boucle parcourt chaque grille stockée dans les DataFrames.
    for idx in range(len(complete_df)):
        complete_grid = complete_df.iloc[idx].values
        incomplete_grid = incomplete_df.iloc[idx].values
#one_hot_encoding
        encoded_complete = [one_hot_encode(int(num)) for num in complete_grid]
        encoded_incomplete = [one_hot_encode(int(num)) for num in incomplete_grid]
#Flattening des Grilles Encodées
        #en une seule liste et ajoutées aux listes X et Y. Cela transforme chaque grille en un vecteur unique qui représente toute la grille, facilitant l'entraînement du modèle de réseau de neurones.


        X.append([bit for number in encoded_incomplete for bit in number])
        Y.append([bit for number in encoded_complete for bit in number])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Définition du modèle de réseau de neurones
class SudokuNN(nn.Module):
    def __init__(self):
        super(SudokuNN, self).__init__()
        #1 couche linaire 27 entres 128 sortie
        self.fc1 = nn.Linear(27, 128)  # Première couche
        #2 128 entres de fills 1 et 64 sortie
        self.fc2 = nn.Linear(128, 64)  # Deuxième couche cachée

        self.fc3 = nn.Linear(64, 27)  # Couche de sortie

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))#fnction dactivation

        x = self.fc3(x).view(-1,9,3)
        x = torch.softmax(x, dim=2)#sortie 3eme couche
        return x
def main():
    complete_path = 'complete_sudoku_grids_3x3.csv'
    incomplete_path = 'incomplete_sudoku_grids_3x3.csv'

    X, Y = load_and_prepare_data(complete_path, incomplete_path)
    #Création du dataset
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))

    #  Le dataset est divisé en deux parties, un set d'entraînement (80% des données) et un set de validation (20% des données).
    train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    model = SudokuNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate for potentially better convergence

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)


            loss = criterion(outputs, labels.view(-1,9,3))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}')

    print("Training complete.")

    # Testing the model with 5 examples from the validation set
    model.eval()
    test_examples = DataLoader(val_dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(test_examples, 1):
        with torch.no_grad():
            example_incomplete, example_complete = data
            example_predicted = model(example_incomplete).squeeze(0)

        print(f"Example {i}:")
        print("Incomplete Sudoku Grid (Input to the model):")
        print(example_incomplete.view(9, 3).argmax(dim=1).view(3, 3).numpy() + 1)
        print("\nComplete Sudoku Grid (Target):")
        print(example_complete.view(9, 3).argmax(dim=1).view(3, 3).numpy() + 1)
        print("\nPredicted Sudoku Grid (Model output):")
        print(example_predicted.view(9, 3).argmax(dim=1).view(3, 3).numpy() + 1)
        print("\n" + "="*50 + "\n")

        if i == 10:  # Display only 5 examples
            break

if __name__ == '__main__':
    main()


'''
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split


def one_hot_encode(number):
    """Converts a number (0-3) to a one-hot encoded vector of size 3."""
    mapping = {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
    return mapping.get(number, [0, 0, 0])


def check_file_existence(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")


def load_and_prepare_data(complete_path, incomplete_path):
    check_file_existence(complete_path)
    check_file_existence(incomplete_path)

    complete_df = pd.read_csv(complete_path)
    incomplete_df = pd.read_csv(incomplete_path)

    X, Y = [], []
    for idx in range(len(complete_df)):
        complete_grid = complete_df.iloc[idx].values
        incomplete_grid = incomplete_df.iloc[idx].values
        encoded_complete = [one_hot_encode(num) for num in complete_grid]
        encoded_incomplete = [one_hot_encode(num) for num in incomplete_grid]
        X.append([bit for number in encoded_incomplete for bit in number])
        Y.append([bit for number in encoded_complete for bit in number])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


class SudokuNN(nn.Module):
    def __init__(self):
        super(SudokuNN, self).__init__()
        self.fc1 = nn.Linear(27, 128)  # input layer to first hidden layer
        self.fc2 = nn.Linear(128, 64)  # first hidden to second hidden layer
        self.fc3 = nn.Linear(64, 27)  # second hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)


def main():
    complete_path = 'complete_sudoku_grids_3x3.csv'
    incomplete_path = 'incomplete_sudoku_grids_3x3.csv'

    X, Y = load_and_prepare_data(complete_path, incomplete_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = SudokuNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.argmax(dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1} completed')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target.argmax(dim=1)).sum().item()

    print(f'Accuracy: {correct / total:.4f}')


if __name__ == '__main__':
    main()
'''
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def one_hot_encode(number):
    """Converts a number (0-3) to a one-hot encoded vector of size 3."""
    mapping = {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
    return np.array(mapping[number], dtype=np.float32)


def load_and_prepare_data(complete_path, incomplete_path):
    """Loads and prepares data from CSV files."""
    complete_df = pd.read_csv(complete_path)
    incomplete_df = pd.read_csv(incomplete_path)

    X = np.array([np.concatenate([one_hot_encode(num) for num in row]) for row in incomplete_df.to_numpy()],
                 dtype=np.float32)
    Y = np.array([np.concatenate([one_hot_encode(num) for num in row]) for row in complete_df.to_numpy()],
                 dtype=np.float32)

    return X, Y


class SudokuNN(nn.Module):
    def __init__(self):
        super(SudokuNN, self).__init__()
        self.fc1 = nn.Linear(27, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 27)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    complete_path = 'complete_sudoku_grids_3x3.csv'
    incomplete_path = 'incomplete_sudoku_grids_3x3.csv'

    X, Y = load_and_prepare_data(complete_path, incomplete_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SudokuNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, 3), target.view(-1, 3).argmax(dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1} completed')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = outputs.view(-1, 3).max(1)
            total += target.view(-1, 3).shape[0]
            correct += (predicted == target.view(-1, 3).argmax(dim=1)).sum().item()

    print(f'Accuracy: {correct / total:.4f}')


if __name__ == '__main__':
    main()
'''
