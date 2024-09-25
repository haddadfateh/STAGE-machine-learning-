import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_graph_data(adjacency_file, color_file):
    # Charge les matrices d'adjacence et les configurations de couleur depuis des fichiers CSV.
    adj_matrices = pd.read_csv(adjacency_file, header=None).values
    adj_matrices = adj_matrices.reshape(-1, 81, 81)  # Reshape en matrices 81x81 pour chaque graphe

    color_configs = pd.read_csv(color_file, header=None).values
    color_configs = color_configs.reshape(-1, 81)  # Reshape en configurations de couleur pour chaque graphe

    return adj_matrices, color_configs

def convert_colors_to_onehot(colors, num_colors=9):
    # Convertit un vecteur de couleurs en encodage one-hot.
    one_hot_colors = np.zeros((len(colors), num_colors))
    for idx, color in enumerate(colors):
        if color > 0:  # Si la couleur est non-nulle
            one_hot_colors[idx, color - 1] = 1
    return one_hot_colors

def create_data_loader(adj_matrices, color_configs):
    # Crée un DataLoader pour traiter les graphes un par un.
    data_list = []
    for adj_matrix, color_config in zip(adj_matrices, color_configs):
        node_features = convert_colors_to_onehot(color_config)
        edge_index = np.array(np.nonzero(adj_matrix))  # Extraction des indices des arêtes

        data = Data(x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long))
        data_list.append(data)

    # Création d'un DataLoader avec un batch size de 1 pour traiter un graphe à la fois
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    return loader
# afiche premier correct
# if __name__ == "__main__":
#     adj_matrices, color_configs = load_graph_data('graph_coloring_adjacency_matrices.csv', 'graph_coloring_color_configs.csv')
#     loader = create_data_loader(adj_matrices, color_configs)
#
#     # Affichage des features et des edge indices pour le premier graphe
#     for data in loader:
#         print("Features (X) for the first graph:")
#         print(data.x)
#         print("\nEdge indices for the first graph:")
#         print(data.edge_index)
#         break  # Afficher uniquement le premier graphe et arrêter


if __name__ == "__main__":
    adj_matrices, color_configs = load_graph_data('graph_coloring_adjacency_matrices.csv', 'graph_coloring_color_configs.csv')
    loader = create_data_loader(adj_matrices, color_configs)

    # Initialiser un compteur pour suivre le nombre de graphes traités
    graph_count = 0

    # Affichage des données pour le deuxième graphe
    for data in loader:
        graph_count += 1  # Incrémenter le compteur à chaque itération
        if graph_count == 2:  # Vérifier si c'est le deuxième graphe
            print("Features (X) for the second graph:")
            print(data.x)
            print("\nEdge indices for the second graph:")
            print(data.edge_index)
            break  # Arrêter après avoir affiché les informations du deuxième graphe


# import torch
# import numpy as np
# import pandas as pd
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
#
# def load_graph_data(adjacency_file, color_file):
#     adj_matrices = pd.read_csv(adjacency_file, header=None).values
#     adj_matrices = adj_matrices.reshape(-1, 81, 81)  # Reshape en une liste de matrices 81x81
#
#     color_configs = pd.read_csv(color_file, header=None).values
#     color_configs = color_configs.reshape(-1, 81)  # Reshape en une liste de configurations de couleur
#
#     return adj_matrices, color_configs
#
# def convert_colors_to_onehot(colors, num_colors=9):
#
#
#     one_hot_colors = np.zeros((len(colors), num_colors))
#     for idx, color in enumerate(colors):
#         if color > 0:  # Si la couleur est non-nulle
#             one_hot_colors[idx, color - 1] = 1
#     return one_hot_colors
# # # Exemple de vecteur de couleurs
# # colors_example = np.array([0, 1, 2, 3, 0, 9, 5])
# #
# # # Application de la fonction
# # one_hot_encoded_colors = convert_colors_to_onehot(colors_example)
# #
# # # Affichage des résultats
# # print("Vecteur de couleurs:", colors_example)
# # print("Encodage one-hot correspondant:\n", one_hot_encoded_colors)
#
# def create_data_loader(adj_matrices, color_configs, batch_size=10):
#
#     data_list = []
#     for adj_matrix, color_config in zip(adj_matrices, color_configs):
#         node_features = convert_colors_to_onehot(color_config)
#         edge_index = np.array(np.nonzero(adj_matrix))  # Extraction des indices des arêtes
#
#         data = Data(x=torch.tensor(node_features, dtype=torch.float),
#                     edge_index=torch.tensor(edge_index, dtype=torch.long))
#         data_list.append(data)
#
#     loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
#     return loader
#
#
#
# # if __name__ == "__main__":
# #     adj_matrices, color_configs = load_graph_data('graph_coloring_adjacency_matrices.csv',
# #                                                   'graph_coloring_color_configs.csv')
# #     loader = create_data_loader(adj_matrices, color_configs, batch_size=10)
# #
# #     # Affichage d'un exemple de données chargées
# #     for data in loader:
# #         print(data)
# #         break  # Afficher seulement le premier lot pour tester
# if __name__ == "__main__":
#     adj_matrices, color_configs = load_graph_data('graph_coloring_adjacency_matrices.csv', 'graph_coloring_color_configs.csv')
#     loader = create_data_loader(adj_matrices, color_configs, batch_size=10)
#     # Retrieve and display data for the first graph in the first batch
#     first_batch = next(iter(loader))  # Get the first batch
#     first_graph_features = first_batch.x[:81]  # Node features for the first graph
#     first_graph_edges = first_batch.edge_index  # Edge indices for the batch
#
#     # Display the node features and edge indices for the first graph
#     print("Features (X) for the first graph:")
#     print(first_graph_features)
#     print("\nEdge indices for the first graph:")
#     print(first_graph_edges[:, first_graph_edges[0] < 81])  # Filter edges for the first graph
