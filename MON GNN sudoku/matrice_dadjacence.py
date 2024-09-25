
import numpy as np
import pandas as pd

def create_sudoku_adjacency_matrix():
    size = 9  # Taille d'une ligne ou d'une colonne dans le Sudoku
    total_cells = size * size  # Total de 81 cellules
    adjacency_matrix = np.zeros((total_cells, total_cells), dtype=int)

    for row in range(size):
        for col in range(size):
            # Index du nœud dans la matrice d'adjacence
            node_index = row * size + col

            # Connecter les lignes et les colonnes
            for i in range(size):
                # Connecter à la même ligne
                adjacency_matrix[node_index, row * size + i] = 1
                # Connecter à la même colonne
                adjacency_matrix[node_index, i * size + col] = 1

            # Connecter le bloc 3x3
            # Trouver le bloc 3x3 auquel le nœud appartient
            start_row = (row // 3) * 3
            start_col = (col // 3) * 3
            for i in range(3):
                for j in range(3):
                    adjacency_matrix[node_index, (start_row + i) * size + (start_col + j)] = 1

            # Lier chaque nœud à lui-même (mettre 1 dans la diagonale)
            adjacency_matrix[node_index, node_index] = 1

    return adjacency_matrix

# Création de la matrice d'adjacence
adjacency_matrix = create_sudoku_adjacency_matrix()

# Convertir en DataFrame pandas
adjacency_df = pd.DataFrame(adjacency_matrix)

# Enregistrer en fichier CSV
file_path = 'sudoku_adjacency_matrix.csv'  # Chemin où vous souhaitez enregistrer le fichier
adjacency_df.to_csv(file_path, index=False)

print(f"Matrice d'adjacence enregistrée avec succès dans {file_path}.")
