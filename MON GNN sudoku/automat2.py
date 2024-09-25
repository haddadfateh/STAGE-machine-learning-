import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

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
        x = F.relu(self.m1(self.conv1(x, edge_index)))
        x = F.relu(self.m2(self.conv2(x, edge_index)))
        x = F.relu(self.m3(self.conv3(x, edge_index)))
        return F.softmax(x, dim=1)

def one_hot_encode(digit):
    encoding = np.zeros(9)
    if 1 <= digit <= 9:
        encoding[digit - 1] = 1
    return encoding

def load_model(model_path):
    model = SudokuGNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prepare_single_data(puzzle, adjacency_matrix):
    node_features = np.array([one_hot_encode(int(num)) for num in puzzle])
    edge_index = np.array([[i, j] for i in range(81) for j in range(81) if adjacency_matrix[i, j] == 1]).T
    data = Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))
    return data

def predict_puzzle(model, puzzle, adjacency_matrix):
    puzzle = list(map(int, puzzle))  # Convert string of numbers into a list of integers
    mask = (np.array(puzzle) > 0).astype(int)  # Create a mask for non-zero entries
    data = prepare_single_data(puzzle, adjacency_matrix)
    data = data.to('cpu')
    output = model(data)
    predictions = output.argmax(dim=1).numpy() + 1
    final_predictions = np.where(mask, puzzle, predictions)
    return final_predictions.reshape(9, 9)

def visualize_sudoku_graph(G, values, title="GRAPHE DU SUDOKO"):
    pos = {i: (i % 9, 8 - i // 9) for i in range(81)}
    labels = {i: str(values.flatten()[i]) for i in range(81)}
    colors = ['#FF6666', '#FFCC66', '#CCFF66', '#66FF66', '#66FFCC', '#66CCFF', '#6666FF', '#CC66FF', '#FF6FCF']
    node_colors = [colors[v - 1] if v > 0 else '#FFFFFF' for v in values.flatten()]

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    plt.title(title)
    plt.axis('off')
    plt.show()

model_path = 'sudoku_gnn_model.pth'
model = load_model(model_path)
adjacency_matrix = pd.read_csv('sudoku_adjacency_matrix.csv', header=None).values
np.fill_diagonal(adjacency_matrix, 0)  # Ensure no self-loops
example_puzzle = "005802347347056200201473695008247539724395108509681024412709856973508012850024970"  # Ensure this is properly formatted as a single string
predicted_solution = predict_puzzle(model, example_puzzle, adjacency_matrix)
G = nx.from_pandas_adjacency(pd.DataFrame(adjacency_matrix), create_using=nx.Graph())
visualize_sudoku_graph(G, predicted_solution, title="Solution Prédite")



#
#
#
# import torch
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
#
# # Définition du modèle GNN pour le Sudoku
# class SudokuGNN(torch.nn.Module):
#     def __init__(self):
#         super(SudokuGNN, self).__init__()
#         self.conv1 = GCNConv(9, 32)
#         self.m1 = torch.nn.BatchNorm1d(32)
#         self.conv2 = GCNConv(32, 32)
#         self.m2 = torch.nn.BatchNorm1d(32)
#         self.conv3 = GCNConv(32, 9)
#         self.m3 = torch.nn.BatchNorm1d(9)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.relu(self.m1(self.conv1(x, edge_index)))
#         x = F.relu(self.m2(self.conv2(x, edge_index)))
#         x = F.relu(self.m3(self.conv3(x, edge_index)))
#         return F.softmax(x, dim=1)
#
# def one_hot_encode(digit):
#     encoding = np.zeros(9)
#     if 1 <= digit <= 9:
#         encoding[digit - 1] = 1
#     return encoding
#
# def load_model(model_path):
#     model = SudokuGNN()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model
#
# def prepare_single_data(puzzle, adjacency_matrix):
#     node_features = np.array([one_hot_encode(int(num)) for num in puzzle])
#     edge_index = np.array([[i, j] for i in range(81) for j in range(81) if adjacency_matrix[i, j] == 1]).T
#     data = Data(x=torch.tensor(node_features, dtype=torch.float),
#                 edge_index=torch.tensor(edge_index, dtype=torch.long))
#     return data
#
# def predict_puzzle(model, puzzle, adjacency_matrix):
#     puzzle = list(map(int, puzzle))  # Convert string of numbers into a list of integers
#     mask = (np.array(puzzle) > 0).astype(int)  # Create a mask for non-zero entries
#     data = prepare_single_data(puzzle, adjacency_matrix)
#     data = data.to('cpu')
#     output = model(data)
#     predictions = output.argmax(dim=1).numpy() + 1
#     # Apply mask to preserve original non-zero entries and only update zeros
#     final_predictions = np.where(mask, puzzle, predictions)
#     return final_predictions.reshape(9, 9)
#
# def visualize_sudoku_graph(G, values, title="GRAPHE DU SUDOKO"):
#     pos = {i: (i % 9, 8 - i // 9) for i in range(81)}
#     labels = {i: str(values.flatten()[i]) for i in range(81)}
#     colors = ['#FF6666', '#FFCC66', '#CCFF66', '#66FF66', '#66FFCC', '#66CCFF', '#6666FF', '#CC66FF', '#FF6FCF']
#     node_colors = [colors[v - 1] if v > 0 else '#FFFFFF' for v in values.flatten()]
#
#     plt.figure(figsize=(8, 8))
#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
#     nx.draw_networkx_edges(G, pos, alpha=0.3)
#     nx.draw_networkx_labels(G, pos, labels, font_size=16)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()
#
# model_path = 'sudoku_gnn_model.pth'
# model = load_model(model_path)
# adjacency_matrix = pd.read_csv('sudoku_adjacency_matrix.csv', header=None).values - np.eye(81)
# example_puzzle = "095802347347056281281473695068247539724395168509681024412739856973508012856124973"  # Ensure this is properly formatted as a single string
# predicted_solution = predict_puzzle(model, example_puzzle, adjacency_matrix)
# G = nx.from_pandas_adjacency(pd.DataFrame(adjacency_matrix), create_using=nx.Graph())
# visualize_sudoku_graph(G, predicted_solution, title="Solution Prédite")
