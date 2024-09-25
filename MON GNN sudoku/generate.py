import numpy as np
import random


def generate_sudoku():
    """ generer sudoko valide . """
    base = 3
    side = base * base

    def pattern(r, c): return (base * (r % base) + r // base + c) % side

    def shuffle(s): return random.sample(s, len(s))

    rBase = range(base)
    rows = [g * base + r for g in shuffle(rBase) for r in shuffle(rBase)]
    cols = [g * base + c for g in shuffle(rBase) for c in shuffle(rBase)]
    nums = shuffle(range(1, base * base + 1))
    board = [[nums[pattern(r, c)] for c in cols] for r in rows]
    return board


def remove_values_from_puzzle(grid, percent=0.1):
    """ filtre. """
    size = 9 * 9  # 81 valeurs dans une grille 9x9
    num_values_to_remove = int(size * percent)
    flat_grid = np.array(grid).flatten()
    indices_to_remove = random.sample(range(size), num_values_to_remove)
    for index in indices_to_remove:
        flat_grid[index] = 0  # Remplacer par 0 pour indiquer une case vide
    return flat_grid.reshape(9, 9)


def main():
    num_puzzles = 30000  # nbr de puzzles a generer

    puzzle_file = open('sudoku_puzzles.csv', 'w')
    solution_file = open('sudoku_solutions.csv', 'w')

    for _ in range(num_puzzles):
        solution = generate_sudoku()
        puzzle = remove_values_from_puzzle(solution, percent=0.1)

        puzzle_str = ''.join(str(num) for num in puzzle.flatten())
        solution_str = ''.join(str(num) for num in np.array(solution).flatten())

        puzzle_file.write(puzzle_str + '\n')
        solution_file.write(solution_str + '\n')

    puzzle_file.close()
    solution_file.close()


if __name__ == "__main__":
    main()
