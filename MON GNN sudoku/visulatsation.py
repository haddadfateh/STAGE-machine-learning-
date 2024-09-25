import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Charger les données depuis les fichiers CSV
adjacency_matrix_path = 'sudoku_adjacency_matrix.csv'  # Mettez à jour le chemin si nécessaire
puzzle_path = 'sudoku_puzzles.csv'
solution_path = 'sudoku_solutions.csv'

adjacency_matrix = pd.read_csv(adjacency_matrix_path, header=None)
puzzles = pd.read_csv(puzzle_path, header=None)
solutions = pd.read_csv(solution_path, header=None)

# Créer un graphe à partir de la matrice d'adjacence
G = nx.from_pandas_adjacency(adjacency_matrix, create_using=nx.Graph)

# Définir une palette de couleurs pour les chiffres de 1 à 9
colors = ['#FF6666', '#FFCC66', '#CCFF66', '#66FF66', '#66FFCC', '#66CCFF', '#6666FF', '#CC66FF', '#FF6FCF']


# Fonction pour visualiser un graphe de Sudoku
def visualize_sudoku_graph(G, values, title="GRAPHE DU SUDOKO"):
    pos = {i: (i % 9, 8 - i // 9) for i in range(81)}  # Position des noeuds en forme de grille 9x9
    labels = {i: values[i] for i in range(81)}
    node_colors = [colors[int(v) - 1] if v != '0' else '#FFFFFF' for v in values]  # Utiliser la palette de couleurs

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    plt.title(title)
    plt.axis('off')  # Cacher les axes pour une meilleure présentation
    plt.show()


# Afficher un exemple de graphes pour un puzzle et sa solution
visualize_sudoku_graph(G, list(puzzles.iloc[0, 0]), title="Puzzle Incomplet")
#visualize_sudoku_graph(G, list(solutions.iloc[0, 0]), title="Solution Complète")
