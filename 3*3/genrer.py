import pandas as pd
import numpy as np
import random

def possible(y, x, n, grid):
    """Vérifie si on peut placer le nombre n en position (y, x) dans la grille 3x3."""
    for i in range(3):
        if grid[y][i] == n or grid[i][x] == n:  # Vérifie ligne et colonne
            return False
    return True

def solve(grid):
    """Tente de résoudre la grille de Sudoku 3x3 en utilisant le backtracking."""
    for y in range(3):
        for x in range(3):
            if grid[y][x] == 0:  # Trouve une case vide
                nums = list(range(1, 4))
                random.shuffle(nums)  # Mélange les nombres pour diversité
                for n in nums:
                    if possible(y, x, n, grid):
                        grid[y][x] = n
                        if solve(grid):
                            return True
                        grid[y][x] = 0
                return False
    return True

def generate_sudoku():
    """Génère une grille de Sudoku 3x3 complète valide."""
    grid = np.zeros((3, 3), dtype=int)
    # Essayer plusieurs fois pour générer une grille initiale diverse
    attempts = 0
    while attempts < 5 and not solve(grid):
        grid = np.zeros((3, 3), dtype=int)
        for i in range(3):
            nums = np.random.choice(range(1, 4), 3, replace=False)
            grid[i, :] = nums  # Initialise des lignes avec des permutations de 1, 2, 3
        attempts += 1
    return grid

def print_grid(grid):
    """Affiche la grille de Sudoku 3x3."""
    for row in grid:
        print(" ".join(str(int(num)) for num in row))
    print()  # Ligne vide pour séparer les grilles

def prepare_data(num_samples=1000):
    """Prépare et sauvegarde les données de Sudoku dans deux fichiers CSV."""
    complete_data = []
    incomplete_data = []

    for _ in range(num_samples):
        np.random.seed()  # Réinitialise la graine aléatoire
        solution = generate_sudoku()
        puzzle = solution.copy()
        mask = np.random.choice([False, True], size=solution.shape, p=[0.5, 0.5])
        puzzle[mask] = 0
        complete_data.append(solution.flatten())
        incomplete_data.append(puzzle.flatten())

    # Convertit les listes en DataFrames
    complete_df = pd.DataFrame(complete_data, columns=[f'cell_{i}' for i in range(9)])
    incomplete_df = pd.DataFrame(incomplete_data, columns=[f'cell_{i}' for i in range(9)])

    # Sauvegarde dans des fichiers CSV
    complete_df.to_csv('complete_sudoku_grids_3x3.csv', index=False)
    incomplete_df.to_csv('incomplete_sudoku_grids_3x3.csv', index=False)
    print("Complete and incomplete datasets saved.")

if __name__ == '__main__':
    prepare_data(1000)
