import pandas as pd
import numpy as np


def print_grid(grid):
    """Fonction pour afficher une grille de Sudoku 3x3."""
    print("\n".join(" ".join(str(int(num)) for num in row) for row in grid))
    print("\n" + "-" * 5 + "\n")


def display_sudoku_grids(csv_file_path, num_grids=5):
    """Fonction pour lire les grilles de Sudoku depuis un fichier CSV et les afficher."""
    df = pd.read_csv(csv_file_path)
    for index, row in df.head(num_grids).iterrows():
        grid = np.array(row).reshape(3, 3)
        print_grid(grid)


if __name__ == '__main__':
    print("Affichage des 5 premières grilles complètes de Sudoku 3x3:")
    display_sudoku_grids('complete_sudoku_grids_3x3.csv')

    print("Affichage des 5 premières grilles incomplètes de Sudoku 3x3:")
    display_sudoku_grids('incomplete_sudoku_grids_3x3.csv')
