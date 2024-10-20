import pygame
import random

# Initialize Pygame
pygame.init()

# Screen size and block settings
BLOCK_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
SCREEN_WIDTH = BOARD_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = BOARD_HEIGHT * BLOCK_SIZE

# Colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Define Tetrimino shapes with colors
SHAPES_WITH_COLORS = [
    ([[1, 1, 1, 1]], CYAN),  # I shape
    ([[1, 1], [1, 1]], YELLOW),  # O shape
    ([[0, 1, 0], [1, 1, 1]], MAGENTA),  # T shape
    ([[0, 1, 1], [1, 1, 0]], GREEN),  # S shape
    ([[1, 1, 0], [0, 1, 1]], RED),  # Z shape
    ([[1, 0, 0], [1, 1, 1]], BLUE),  # J shape
    ([[0, 0, 1], [1, 1, 1]], ORANGE)  # L shape
]

def create_board():
    return [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]

# Updated function to draw a block with a black border
def draw_block(screen, color, position):
    x, y = position[0] * BLOCK_SIZE, position[1] * BLOCK_SIZE

    # Draw black border
    pygame.draw.rect(screen, BLACK, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))

    # Draw inner colored block
    inner_margin = 2  # Thickness of the black border
    pygame.draw.rect(
        screen, color,
        pygame.Rect(
            x + inner_margin, y + inner_margin,
            BLOCK_SIZE - 2 * inner_margin, BLOCK_SIZE - 2 * inner_margin
        )
    )

def draw_board(screen, board):
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if board[y][x]:
                draw_block(screen, GRAY, (x, y))

def draw_shape(screen, shape, color, position):
    for y, row in enumerate(shape):
        for x, value in enumerate(row):
            if value:
                draw_block(screen, color, (x + position[0], y + position[1]))

def valid_position(board, shape, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                if (x + off_x < 0 or x + off_x >= BOARD_WIDTH or
                    y + off_y >= BOARD_HEIGHT or
                    board[y + off_y][x + off_x]):
                    return False
    return True

def add_shape_to_board(board, shape, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, value in enumerate(row):
            if value:
                board[off_y + y][off_x + x] = 1

def clear_lines(board):
    new_board = [row for row in board if any(cell == 0 for cell in row)]
    lines_cleared = BOARD_HEIGHT - len(new_board)
    new_board = [[0] * BOARD_WIDTH for _ in range(lines_cleared)] + new_board
    return new_board, lines_cleared

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Tetris')

    clock = pygame.time.Clock()
    board = create_board()

    current_shape, current_color = random.choice(SHAPES_WITH_COLORS)
    shape_pos = [BOARD_WIDTH // 2 - len(current_shape[0]) // 2, 0]

    game_over = False
    fall_speed = 500
    last_fall_time = pygame.time.get_ticks()

    while not game_over:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    shape_pos[0] -= 1
                    if not valid_position(board, current_shape, shape_pos):
                        shape_pos[0] += 1
                if event.key == pygame.K_RIGHT:
                    shape_pos[0] += 1
                    if not valid_position(board, current_shape, shape_pos):
                        shape_pos[0] -= 1
                if event.key == pygame.K_DOWN:
                    shape_pos[1] += 1
                    5*fall_speed
                    if not valid_position(board, current_shape, shape_pos):
                        shape_pos[1] -= 1
                if event.key == pygame.K_UP:
                    rotated_shape = list(zip(*current_shape[::-1]))
                    if valid_position(board, rotated_shape, shape_pos):
                        current_shape = rotated_shape

        current_time = pygame.time.get_ticks()
        if current_time - last_fall_time > fall_speed:
            shape_pos[1] += 1
            last_fall_time = current_time

            if not valid_position(board, current_shape, shape_pos):
                shape_pos[1] -= 1
                add_shape_to_board(board, current_shape, shape_pos)
                board, _ = clear_lines(board)
                current_shape, current_color = random.choice(SHAPES_WITH_COLORS)
                shape_pos = [BOARD_WIDTH // 2 - len(current_shape[0]) // 2, 0]

                if not valid_position(board, current_shape, shape_pos):
                    game_over = True

        screen.fill(BLACK)
        draw_board(screen, board)
        draw_shape(screen, current_shape, current_color, shape_pos)
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
