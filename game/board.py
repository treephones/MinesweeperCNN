import random

import numpy as np
import pygame

import tensorflow

MINE_CONST = 100
COVERED_CONST = -1

# Reveal square at y, x in cur_grid
def reveal_square(y, x, cur_grid, uncovered_grid):
    size = get_dimensions(cur_grid)
    
    # The selected square is a mine and player loses
    if (uncovered_grid[y][x] == -1):
        return False
    
    # If there is a mine nearby, uncover the selected square
    elif (uncovered_grid[y][x] > 0):
        cur_grid[y][x] = uncovered_grid[y][x]
        return cur_grid

    # The current square has no mines around it and perform BFS to uncover all connected squares of 0 mines
    else:
        
        # Uncover this square and search all squares around it
        cur_grid[y][x] = uncovered_grid[y][x]
        toSearch = [(y, x)]
        
        # While toSearch has elements
        while (toSearch):
            
            # Pop the front element and search all points around it
            y, x = toSearch.pop(0)
            for coord in surrounding_points(y, x):
                    
                new_y, new_x = coord
                # If a square is in grid boundaries, not a mine, and covered, reveal square
                if new_y < size[0] and new_y >= 0 and new_x < size[1] and new_x >= 0 and uncovered_grid[new_y][new_x] != -1 and cur_grid[new_y][new_x] == "#":
                    cur_grid[new_y][new_x] = uncovered_grid[new_y][new_x]
                    
                    # If this position has 0 mines around it, add to toSearch
                    if (uncovered_grid[new_y][new_x] == 0):
                        toSearch.append((new_y, new_x))                

        return cur_grid


# Flag square at (y, x)
def flag_square(y, x, cur_grid):
    
    # If the element at this position isn't a number and is uncovered, flag it
    if isinstance(cur_grid[y][x], str) and cur_grid[y][x] == "#":
        cur_grid[y][x] = "!"
        
    # If the element at this position isn't a number and is flagged, unflag it
    elif isinstance(cur_grid[y][x], str) and cur_grid[y][x] == "!":
        cur_grid[y][x] = "#"
    return cur_grid

# Calculate mines remaining in cur_grid
def mines_remaining(cur_grid, uncovered_grid):
    return sum(element == -1 for row in uncovered_grid for element in row) - sum(
        element == '!' for row in cur_grid for element in row)

# Clear covered squares if number of flags around square == number on square
def clear_square(y, x, cur_grid, uncovered_grid):
    if (get_surroundings(y, x, cur_grid).count("!") == cur_grid[y][x]):
        size = get_dimensions(cur_grid)

        # For every point around (y, x)
        for coord in surrounding_points(y, x):
            coord_y, coord_x = coord
            
            # If within bounds and not flagged a mine
            if coord_y < size[0] and coord_y >= 0 and coord_x < size[1] and coord_y >= 0 and cur_grid[coord_y][coord_x] != "!":
                cur_grid = reveal_square(coord_y, coord_x, cur_grid, uncovered_grid)
                if not cur_grid:
                    return False
    return cur_grid

# Check if all of the flagged mines are in the right spot
def check_if_correct(cur_grid, uncovered_grid):
    for i in range(len(cur_grid)):
        for j in range(len(cur_grid[0])):
            # If flagged and is actually a mine
            if cur_grid[i][j] == "!" and uncovered_grid[i][j] != -1:
                return False
            
    # True means player won
    return True


# Generating data for model
def generateForData(n, m, n_mines):
    
    # Make empty grid
    grid = [[0] * m for _ in range(n)]
    mines = set()
    # Adding mines
    while len(mines) < n_mines:
        y = random.randint(0, n - 1)
        x = random.randint(0, m - 1)
        if (y, x) not in mines:
            mines.add((y, x))
            grid[y][x] = MINE_CONST  # -1
            
    # Filling numbers on grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != MINE_CONST:
                surroundings = get_surroundings(i, j, grid)
                grid[i][j] = surroundings.count(MINE_CONST)
    return grid

# Get surrounding points of y and x
def surrounding_points(y, x):
    return [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1), (y + 1, x + 1), (y - 1, x - 1), (y + 1, x - 1),
            (y - 1, x + 1)]

# Get labels of surrounding points of a point on the grid
def get_surroundings(y, x, grid):
    values = []
    for point in surrounding_points(y, x):
        if point[0] >= 0 and point[1] >= 0:
            try:
                values.append(grid[point[0]][point[1]])
            except IndexError:
                continue
    return values

# Generate a board of size n by m with n_mines mines and player starting at coord
def generate(n, m, n_mines, coord):
    coord_y, coord_x = coord
    
    # Make grid full of zeros
    grid = [[0] * m for _ in range(n)]
    mines = set()
    
    # Get neighbours of coord_y, coord_x which is where the player chooses to start
    neighbours = surrounding_points(coord_y, coord_x)
    
    # Picking where mines are randomly
    while len(mines) < n_mines:
        y = random.randint(0, n - 1)
        x = random.randint(0, m - 1)
        
        # If (y, x) is not already a mine and is not where the player chooses to start or one of its neighbours, add it as a mine
        if (y, x) not in mines and (y, x) != coord and (y, x) not in neighbours:
            mines.add((y, x))
            grid[y][x] = -1

    # Rect the size of the game board
    rect = pygame.Rect(10, 10, cell_size * columns, cell_size * rows)
    pygame.draw.rect(screen, (220, 220, 220), rect)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            
            # Making squares for each square on the grid
            rect = pygame.Rect(10 + i * cell_size, 10 + j * cell_size, cell_size + 1, cell_size + 1)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)
            
            # Making squares of the heatmap
            rect = pygame.Rect(770 + i * cell_size, 10 + j * cell_size, cell_size + 1, cell_size + 1)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)

            # If not a mine, set the spot in the grid to the number of mines around it
            if grid[i][j] != -1:
                surroundings = get_surroundings(i, j, grid)
                grid[i][j] = surroundings.count(-1)
    pygame.display.flip()
    return grid

# Generate a covered board of size n by m
def generate_covered(n, m):
    grid = [["#"] * m for _ in range(n)]
    return grid

# Creating a randomly covered grid for model
def random_coverage(grid):
    covered_grid = generate_covered(len(grid), len(grid[0]))
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != MINE_CONST:
                if random.choice([1, 2]) == 1:
                    covered_grid = reveal_square(i, j, covered_grid, grid)
            else:
                if random.randint(0, 5) in [1]:
                    covered_grid[i][j] = MINE_CONST
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if covered_grid[i][j] == '#':
                covered_grid[i][j] = COVERED_CONST
    return covered_grid

# Creating the grid with labels for model
def create_label_grid(grid):
    label_grid = [[0] * len(grid[0]) for _ in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == MINE_CONST:
                label_grid[i][j] = 1
    return label_grid

# Create new covered grid from covered_grid
def create_coverage_mask(covered_grid):
    new_covered_grid = covered_grid.copy()
    for i in range(len(covered_grid)):
        for j in range(len(covered_grid[0])):
            if covered_grid[i][j] == COVERED_CONST:
                new_covered_grid[i][j] = 1
            else:
                new_covered_grid[i][j] = 0
    return new_covered_grid

# Creating data for model using above functions
def create_data(n, m, n_mines, amount):
    boards = []
    # Amount boards
    for i in range(amount):
        print(f'generated data point {i + 1}/{amount}')
        grid = generateForData(n, m, n_mines)
        label_grid = create_label_grid(grid)
        grid = random_coverage(grid)
        
        coverage_map = create_coverage_mask(grid)
        
        grid = np.array(grid)
        grid = np.expand_dims(grid, axis=-1)
        grid = np.expand_dims(grid, axis=-0)

        label_grid = np.array(label_grid)
        label_grid = np.expand_dims(label_grid, axis=-1)
        label_grid = np.expand_dims(label_grid, axis=-0)

        coverage_map = np.array(coverage_map)
        coverage_map = np.expand_dims(coverage_map, axis=-1)
        coverage_map = np.expand_dims(coverage_map, axis=-0)

        boards.append([grid, label_grid, coverage_map])
    return boards

# Set dimension of matrix
def set_dimensions(matrix):
    matrix = np.array(matrix)
    matrix = np.expand_dims(matrix, axis=-1)
    matrix = np.expand_dims(matrix, axis=-0)
    return matrix

# Get dimensions of grid
def get_dimensions(grid):
    return (len(grid), len(grid[0]))

# Display grid probabilities to cur_grif
def disp_grid_to_model_grid(cur_grid):
    grid = [[0]*len(cur_grid[0]) for _ in range(len(cur_grid))]
    for i in range(len(cur_grid)):
        for j in range(len(cur_grid[0])):
            if (cur_grid[i][j] == '#'):
                grid[i][j] = -1
            elif (cur_grid[i][j] == '!'):
                grid[i][j] = 100
            else:
                grid[i][j] = int(cur_grid[i][j])
    return grid

# Update board with model
def update_board(cur_grid, model):
    size = get_dimensions(cur_grid)
    mgrid = disp_grid_to_model_grid(cur_grid)
    mgrid = set_dimensions(mgrid)
    
    # Get predictions from model
    ans = model.predict(mgrid)
    ans = [[round(float(val), 2) for val in row] for row in ans.squeeze()]
    
    # Sets to save what is safe and a mine
    safe = set()
    mine = set()
    for i in range(len(cur_grid)):
        for j in range(len(cur_grid[0])):
            # Make a rect for square on grid for grid and heatmap
            rect = pygame.Rect(10 + i * cell_size, 10 + j * cell_size, cell_size + 1, cell_size + 1)
            rect2 = pygame.Rect(770 + i * cell_size, 10 + j * cell_size, cell_size + 1, cell_size + 1)
            
            # Colour if its covered
            if (cur_grid[i][j] == '#'):
                pygame.draw.rect(screen, (200, 200, 200), rect)
                pygame.draw.rect(screen, (220*ans[i][j], 0, 0), rect2)

            # Colour if its 0
            elif (cur_grid[i][j] == 0):
                pygame.draw.rect(screen, (255, 255, 255), rect)
                pygame.draw.rect(screen, (255, 255, 255), rect2)
                
            # Colour if it's a mine
            elif (cur_grid[i][j] == '!'):
                pygame.draw.rect(screen, (220, 0, 0), rect)
                pygame.draw.rect(screen, (220*ans[i][j], 0, 0), rect2)
                
            # Number on point in grid
            else:
                
                # If number of flagged around grid is number of mines around square
                if (get_surroundings(i, j, cur_grid).count("!") == cur_grid[i][j]):
                    
                    # For each of its neighbours
                    for coord in surrounding_points(i, j):
                        coord_y, coord_x = coord
                        
                        # Add to safe since its guaranteed to be safe if its not flagged
                        if coord_y < size[0] and coord_y >= 0 and coord_x < size[1] and coord_x >= 0 and cur_grid[coord_y][coord_x] == '#' and (coord_y, coord_x) not in safe:
                            safe.add((coord_y, coord_x))
                # If number of covered squares plus number of flagged are equal to number on square
                elif (get_surroundings(i, j, cur_grid).count("#") + get_surroundings(i, j, cur_grid).count("!") == cur_grid[i][j]):
                    
                    # Visit each of its neighbours
                    for coord in surrounding_points(i, j):
                        coord_y, coord_x = coord
                        
                        # Add to mines since its guaranteed to be mine if its covered
                        if coord_y < size[0] and coord_y >= 0 and coord_x < size[1] and coord_x >= 0 and cur_grid[coord_y][coord_x] == '#' and (coord_y, coord_x) not in mine:
                            mine.add((coord_y, coord_x))
                # Draw squares as white
                pygame.draw.rect(screen, (255, 255, 255), rect)
                pygame.draw.rect(screen, (255, 255, 255), rect2)
                
                # Show mine count
                screen.blit(my_font.render(str(get_surroundings(i, j, uncovered_grid).count(-1)), False, (0, 0, 0)),
                            (i * cell_size + 15, j * cell_size))
            # Outline of square
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
            pygame.draw.rect(screen, (0, 0, 0), rect2, 1)
            
    # Make each guaranteed safe square white on heatmap
    for coord in safe:
        i, j = coord
        rect2 = pygame.Rect(770 + i * cell_size, 10 + j * cell_size, cell_size + 1, cell_size + 1)
        pygame.draw.rect(screen, (255, 255, 255), rect2)
        pygame.draw.rect(screen, (0, 0, 0), rect2, 1)
        
    # Make each guaranteed mine square red on heatmap
    for coord in mine:
        i, j = coord
        rect2 = pygame.Rect(770 + i * cell_size, 10 + j * cell_size, cell_size + 1, cell_size + 1)
        pygame.draw.rect(screen, (220, 0, 0), rect2)
        pygame.draw.rect(screen, (0, 0, 0), rect2, 1)
        
    pygame.display.flip()


if __name__ == '__main__':
    rows = 16
    columns = 30
    mines = 99
    height = 600
    width = 1600
    cell_size = (width - 200) / (columns * 2)
    
    # Generate grid on first iteration
    generate_grid = True
    grid = generate_covered(columns, rows)
    
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    screen.fill((255, 255, 255))
    pygame.font.init()
    my_font = pygame.font.SysFont('Comic Sans MS', 30)

    model = tensorflow.keras.models.load_model("../model/bestsofarcomplex.keras")

    # Make starting grid so user can pick where to start
    for i in range(len(grid)):
        for j in range(len(grid[0])):

            rect = pygame.Rect(10 + i * cell_size, 10 + j * cell_size, cell_size + 1, cell_size + 1)
            rect2 = pygame.Rect(770 + i * cell_size, 10 + j * cell_size, cell_size + 1, cell_size + 1)
            pygame.draw.rect(screen, (200, 200, 200), rect)
            pygame.draw.rect(screen, (200, 200, 200), rect2)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
            pygame.draw.rect(screen, (0, 0, 0), rect2, 1)
    pygame.display.flip()

    cur_grid = []
    while isinstance(cur_grid, list):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                screen.fill((255, 255, 255))

                x, y = pygame.mouse.get_pos()

                row = int((y - 10) // cell_size)
                col = int((x - 10) // cell_size)
                
                # If click is not in bounds
                if row > rows or row < 0 or col > columns or columns < 0:
                    break

                # Left click
                if event.button == 1:
                    
                    # Generate grid and make sure where user starts is safe
                    if generate_grid:
                        uncovered_grid = generate(columns, rows, mines, (col, row))
                        cur_grid = generate_covered(columns, rows)
                        label_grid = create_label_grid(uncovered_grid)
                        generate_grid = False
                    
                    # If the spot is covered, reveal square
                    if cur_grid[col][row] == '#':
                        cur_grid = reveal_square(col, row, cur_grid, uncovered_grid)
                        
                    # Clear surrounding squares if it is uncovered
                    else:
                        cur_grid = clear_square(col, row, cur_grid, uncovered_grid)

                # Flag if right click
                elif event.button == 3:
                    cur_grid = flag_square(col, row, cur_grid)

                # False if they lose
                if cur_grid == False:
                    break
                
                # Update board
                update_board(cur_grid, model)

                
                rect = pygame.Rect(770 + columns * cell_size, 0, cell_size + 1, cell_size + 1)
                pygame.draw.rect(screen, (255, 255, 255), rect)
                # Display mines left
                screen.blit(my_font.render(str(mines_remaining(cur_grid, uncovered_grid)) + " left", False, (0, 0, 0)),
                            ((columns - 4) * cell_size, (rows + 1) * cell_size))

                # Check if correct if no mines left
                if mines_remaining(cur_grid, uncovered_grid) == 0 and check_if_correct(cur_grid, uncovered_grid):
                    cur_grid = True
                    break
                pygame.display.flip()
                
    # See if user won
    if cur_grid:
        print("You won")
    else:
        print("You lost")

