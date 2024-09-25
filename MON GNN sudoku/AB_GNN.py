import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# chrg  les donneees
def load_data():
    adjacency_matrix = pd.read_csv('sudoku_adjacency_matrix.csv', header=None).values
    puzzles = pd.read_csv('sudoku_puzzles.csv', header=None).values
    solutions = pd.read_csv('sudoku_solutions.csv', header=None).values
    return adjacency_matrix, puzzles, solutions

# One-hot encoding pour les chiffres
def one_hot_encode(digit):
    encoding = np.zeros(9)
    if digit > 0:
        encoding[digit - 1] = 1
    return encoding

# Préparation des données pour le GNN
def prepare_data(puzzles, solutions, adjacency_matrix):
    data_list = []
    for puzzle, solution in zip(puzzles, solutions):
        node_features = np.array([one_hot_encode(int(num)) for num in puzzle[0]])
        labels = np.array([int(num) - 1 for num in solution[0]])
        edge_index = np.array([[i, j] for i in range(81) for j in range(81) if adjacency_matrix[i, j] == 1]).T

        data = Data(x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    y=torch.tensor(labels, dtype=torch.long))
        data_list.append(data)
    return data_list

# Définition du modèle GNN pour le Sudoku
class SudokuGNN(torch.nn.Module):
    def __init__(self):
        super(SudokuGNN, self).__init__()
        self.conv1 = GCNConv(9, 32)

        self.m1 = torch.nn.BatchNorm1d(32)

        self.conv2 = GCNConv(32, 32)

        self.m2 = torch.nn.BatchNorm1d(32)

        self.conv3 = GCNConv(32, 9)

        self.m3 = torch.nn.BatchNorm1d(9)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(x.size())

        x = F.relu(self.m1(self.conv1(x, edge_index)))

        x = F.relu(self.m2(self.conv2(x, edge_index)))

        x = F.relu(self.m3(self.conv3(x, edge_index)))


        return F.softmax(x, dim=1)


metric = "nb_conflicts"
#metric ="accuracy_solution"

# Fonctions pour entrnmnt et evaluation
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        # print("data.size()")
        # print(data)
        # print(data.num_graphs)

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        mask_input = torch.sum(data.x, 1).unsqueeze(1)# le masque la case deja defini et la cose no defini
        # Résultat : tensor([[1], [1], [0]])

        output_with_input_digit = data.x * mask_input + (1 - mask_input)* output
        #predire que les case vide et recuprer les prediction des case vide

        if(metric == "accuracy_solution"):
            loss = criterion(output_with_input_digit, data.y)
            # output_with_input_digit :est la sortie du modèle combinée avec les chiffres initiaux.

        elif(metric == "nb_conflicts"):
#data.edge_index contient les indices des aretes du graphe.
#index1 et index2 :sont les indices des nœuds connectés par les arêtes.
            index1 = data.edge_index[0,:]
            index2 = data.edge_index[1,:]
#    Par exemple, si une arête relie les nœuds 3 et 7, index1 serait 3 et index2 serait 7.



            loss = (output_with_input_digit[index1]*output_with_input_digit[index2]).sum()
            #output_with_input_digit[index1] récupère les prédictions pour les nœuds dans index1.
#output_with_input_digit[index2] récupère les prédictions pour les nœuds dans index2


        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0  #__nbr de pred corrct
    total_loss = 0# perte total de lensemble des donnééees
    total_nodes = 0# nombre total des noeuds predis dans le batch

    nb_conflit = 0
    total_edges = 0

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            output = model(data)

            mask_input = torch.sum(data.x, 1).unsqueeze(1)# Identifier les cellules déjà remplies dans le puzzle
            output_with_input_digit = data.x * mask_input + (1 - mask_input) * output
            #Combine la sortie du modèle avec l'état initial du puzzle, en conservant les chiffres initiaux.


            if (metric == "accuracy_solution"):

                loss = criterion(output_with_input_digit, data.y)
                total_loss += loss.item()#ajoute la perte de ce batch à la perte totale

                pred = torch.argmax(output_with_input_digit, 1)# Obtenir les prédictions avec les plus hautes probabilités

                correct += pred.eq(data.y).sum().item()# Compter les prédictions correctes
                total_nodes += data.y.size(0)# Compter le nombre total de nœuds

            elif (metric == "nb_conflicts"):

                index1 = data.edge_index[0, :]
                index2 = data.edge_index[1, :]

                loss = (output_with_input_digit[index1] * output_with_input_digit[index2]).sum()#Calcule la somme des conflits pour chaque arête.
                total_loss += loss.item()

                pred = torch.argmax(output_with_input_digit, 1)#Prédit le chiffre avec la plus haute probabilité pour chaque cellule.
                zeros = torch.zeros(output_with_input_digit.size()).to(device)
                ones = torch.ones(output_with_input_digit.size()).to(device)#
                onehot_pred = torch.scatter(zeros, 1, pred.unsqueeze(1), ones)#Convertit les prédictions en format one-hot encodé

                nb_conflit += ((onehot_pred[index1] * onehot_pred[index2]).sum()).item()

                total_edges += data.edge_index.shape[1]

    avg_loss = total_loss / len(loader)
    if(metric == "accuracy_solution"):
        accuracy = correct / total_nodes

        return accuracy, avg_loss

    elif(metric == "nb_conflicts"):

        pr_conflict_edges = nb_conflit / total_edges

        return pr_conflict_edges, avg_loss





# charge  les données et preparationn des DataLoader
adjacency_matrix, puzzles, solutions = load_data()

adjacency_matrix = adjacency_matrix - np.eye(81)



data_list = prepare_data(puzzles, solutions, adjacency_matrix)
train_data, val_data = data_list[:25000], data_list[25000:]

#
# data_list = prepare_data(puzzles[:20], solutions[:20], adjacency_matrix)
# train_data, val_data = data_list[:10], data_list[10:]



# print(train_data)

train_loader = DataLoader(train_data, batch_size=500, shuffle=True)#batch a 500 grand je pens
val_loader = DataLoader(val_data, batch_size=500, shuffle=False)

# init  du modele et des param d'dentrainement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SudokuGNN().to(device)

#9) Afficher le nombre de paramètres de chaque réseau
#print(model.parameters().n)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

#  entrainement et evaluation
train_losses, val_losses, val_accuracies = [], [], []
# for epoch in range(100000):
for epoch in range(150):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_accuracy, val_loss = evaluate(model, val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    if(metric == "accuracy_solution"):
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    elif(metric == "nb_conflicts"):
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Pr edge conflict: {val_accuracy:.4f}')


# Affichage des courbes de perte et de précision
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
