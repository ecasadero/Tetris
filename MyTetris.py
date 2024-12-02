import random
import pygame
import sqlite3



# Initialize Pygame

pygame.init()



# Screen size and block settings

BLOCK_SIZE = 30  # 30 pixels

BOARD_WIDTH = 10  # Width and Height are in block sizes

BOARD_HEIGHT = 20

SCREEN_WIDTH = BOARD_WIDTH * BLOCK_SIZE

SCREEN_HEIGHT = BOARD_HEIGHT * BLOCK_SIZE

SIDE_PANEL_WIDTH = 200



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

SHAPES_WITH_COLORS = {

    "IShape": ([[1, 1, 1, 1]], CYAN),  # I shape

    "OShape": ([[1, 1], [1, 1]], YELLOW),  # O shape

    "TShape": ([[0, 1, 0], [1, 1, 1]], MAGENTA),  # T shape

    "SShape": ([[0, 1, 1], [1, 1, 0]], GREEN),  # S shape

    "ZShape": ([[1, 1, 0], [0, 1, 1]], RED),  # Z shape

    "JShape": ([[1, 0, 0], [1, 1, 1]], BLUE),  # J shape

    "LShape": ([[0, 0, 1], [1, 1, 1]], ORANGE),  # L shape

}



def initialize_database():

    conn = sqlite3.connect('tetris_stats.db')

    c = conn.cursor()



    # Create the table if it doesn't already exist

    c.execute('''

        CREATE TABLE IF NOT EXISTS game_stats (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            piece_type TEXT,

            occurrences INTEGER,

            single_lines INTEGER,

            double_lines INTEGER,

            triple_lines INTEGER,

            tetris_lines INTEGER,

            total_lines INTEGER

        )

    ''')



    # Insert initial records for all tetromino types if not already present

    tetrominoes = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']

    for piece in tetrominoes:

        c.execute('''

            INSERT OR IGNORE INTO game_stats (piece_type, occurrences, single_lines, double_lines, triple_lines, tetris_lines, total_lines)

            VALUES (?, 0, 0, 0, 0, 0, 0)

        ''', (piece,))



    conn.commit()

    conn.close()



def record_tetromino(piece_type):

    conn = sqlite3.connect('tetris_stats.db')

    c = conn.cursor()

    c.execute('''

        UPDATE game_stats

        SET occurrences = occurrences + 1

        WHERE piece_type = ?

    ''', (piece_type,))

    conn.commit()

    conn.close()



def create_board():

    return [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]



# Updated function to draw a block with a black border

def draw_block(screen, color, position):

    x, y = position[0] * BLOCK_SIZE, position[1] * BLOCK_SIZE  # Use this coordinate to know where to draw the borders

    pygame.draw.rect(screen, BLACK, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))  # Draws the black border

    inner_margin = 2  # Thickness of the black border

    pygame.draw.rect(screen, color, pygame.Rect(

        x + inner_margin, y + inner_margin,

        BLOCK_SIZE - 2 * inner_margin, BLOCK_SIZE - 2 * inner_margin  # Draws the colored block with an offset for the border

    ))



def washed_out_color(color):  # Calculate the washed-out version of a color.

    r, g, b = color

    return (

        (r + 255) // 2,

        (g + 255) // 2,

        (b + 255) // 2

    )



def add_shape_to_board(board, shape, color, offset, shape_type):

    off_x, off_y = offset

    for y, row in enumerate(shape):

        for x, value in enumerate(row):

            if value:

                board[off_y + y][off_x + x] = color  # Store the block's color



    # Record the placement of the tetromino

    record_tetromino(shape_type)



def draw_board(screen, board):

    for y in range(BOARD_HEIGHT):

        for x in range(BOARD_WIDTH):

            if board[y][x]:  # Check if there's a color stored in this cell

                faded_color = washed_out_color(board[y][x])  # Get washed-out color

                draw_block(screen, faded_color, (x, y))



def record_line_clears(lines_cleared, piece_type):

    conn = sqlite3.connect('tetris_stats.db')

    c = conn.cursor()



    # Update total lines cleared

    c.execute('''

        UPDATE game_stats

        SET total_lines = total_lines + ?

        WHERE piece_type = ?

    ''', (lines_cleared, piece_type))



    # Update specific line clear counts

    if lines_cleared == 1:

        c.execute('UPDATE game_stats SET single_lines = single_lines + 1 WHERE piece_type = ?', (piece_type,))

    elif lines_cleared == 2:

        c.execute('UPDATE game_stats SET double_lines = double_lines + 1 WHERE piece_type = ?', (piece_type,))

    elif lines_cleared == 3:

        c.execute('UPDATE game_stats SET triple_lines = triple_lines + 1 WHERE piece_type = ?', (piece_type,))

    elif lines_cleared == 4:

        c.execute('UPDATE game_stats SET tetris_lines = tetris_lines + 1 WHERE piece_type = ?', (piece_type,))



    conn.commit()

    conn.close()



def shape_matches(shape1, shape2):

    """Check if two shapes are the same (ignoring format differences)."""

    return [list(row) for row in shape1] == [list(row) for row in shape2]



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



def clear_lines(board):

    new_board = [row for row in board if any(cell == 0 for cell in row)]

    lines_cleared = BOARD_HEIGHT - len(new_board)

    new_board = [[0] * BOARD_WIDTH for _ in range(lines_cleared)] + new_board

    return new_board, lines_cleared

def draw_side_panel(screen, score, level, next_shape_data, held_shape_data):
        font = pygame.font.SysFont("Times New Roman", 24)

        # Draw Score
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (SCREEN_WIDTH + 10, SCREEN_HEIGHT // 2 - 80))

        # Draw Level
        level_text = font.render(f"Level: {level}", True, WHITE)
        screen.blit(level_text, (SCREEN_WIDTH + 10, SCREEN_HEIGHT // 2 - 5))

        # Draw Next Shape Preview
        next_text = font.render("Next:", True, WHITE)
        screen.blit(next_text, (SCREEN_WIDTH + 10, 30))
        if next_shape_data:
            next_shape, next_color = next_shape_data
            for y, row in enumerate(next_shape):
                for x, value in enumerate(row):
                    if value:
                        draw_block(screen, next_color, (x + BOARD_WIDTH + 1, y + 2))

        # Draw Held Shape
        hold_text = font.render("Hold:", True, WHITE)
        screen.blit(hold_text, (SCREEN_WIDTH + 10, SCREEN_HEIGHT - 200))
        if held_shape_data:
            held_shape, held_color = held_shape_data
            for y, row in enumerate(held_shape):
                for x, value in enumerate(row):
                    if value:
                        draw_block(screen, held_color, (x + BOARD_WIDTH + 1, y + BOARD_HEIGHT - 5))


def main():
    initialize_database()
    screen = pygame.display.set_mode((SCREEN_WIDTH + SIDE_PANEL_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Tetris')

    clock = pygame.time.Clock()
    board = create_board()

    SHAPES_LIST = list(SHAPES_WITH_COLORS.items())
    current_shape_name, (current_shape, current_color) = random.choice(SHAPES_LIST)
    next_shape_name, (next_shape, next_color) = random.choice(SHAPES_LIST)
    held_shape = None
    hold_used = False
    shape_pos = [BOARD_WIDTH // 2 - len(current_shape[0]) // 2, 0]
    score = 0

    total_lines_cleared = 0
    level = 0
    fall_speed = 500  # Initial fall speed in milliseconds
    last_fall_time = pygame.time.get_ticks()
    game_over = False

    MIN_FALL_SPEED = 50  # Minimum fall speed in milliseconds

    soft_drop_speed = 50  # Time in milliseconds between soft drop movements
    soft_drop_time = pygame.time.get_ticks()

    while not game_over:
        dt = clock.tick(60)

        # Get the current state of all keys
        keys = pygame.key.get_pressed()

        # Event handling
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
                    if not valid_position(board, current_shape, shape_pos):
                        shape_pos[1] -= 1
                if event.key == pygame.K_UP:
                    rotated_shape = [list(row) for row in zip(*current_shape[::-1])]
                    if valid_position(board, rotated_shape, shape_pos):
                        current_shape = rotated_shape
                if event.key == pygame.K_c and not hold_used:
                    if held_shape is None:
                        held_shape = (current_shape, current_color)
                        current_shape_name, (current_shape, current_color) = next_shape_name, (next_shape, next_color)
                        next_shape_name, (next_shape, next_color) = random.choice(SHAPES_LIST)
                    else:
                        # Swap held and current shapes
                        held_shape, (current_shape, current_color) = (current_shape, current_color), held_shape
                    shape_pos = [BOARD_WIDTH // 2 - len(current_shape[0]) // 2, 0]
                    hold_used = True

        # Get the current time
        current_time = pygame.time.get_ticks()

        # Handle soft drop when down key is held
        if keys[pygame.K_DOWN]:
            if current_time - soft_drop_time > soft_drop_speed:
                shape_pos[1] += 1
                soft_drop_time = current_time
                if not valid_position(board, current_shape, shape_pos):
                    shape_pos[1] -= 1
                    # Handle locking and generating new piece
                    # ... [Existing code for locking and new piece generation] ...

        # Timing code for automatic fall
        if current_time - last_fall_time > fall_speed:
            shape_pos[1] += 1
            last_fall_time = current_time

            if not valid_position(board, current_shape, shape_pos):
                shape_pos[1] -= 1
                # Add the shape to the board using the known shape name
                add_shape_to_board(board, current_shape, current_color, shape_pos, current_shape_name)

                # Clear lines and get the number of lines cleared
                board, lines_cleared = clear_lines(board)

                # Record line clears using the known shape name
                if lines_cleared > 0:
                    record_line_clears(lines_cleared, current_shape_name)

                # Update the score
                score += lines_cleared * 100

                # Update total lines cleared
                total_lines_cleared += lines_cleared

                # Check if it's time to increase the level
                new_level = total_lines_cleared // 10
                if new_level > level:
                    level = new_level
                    fall_speed = int(fall_speed * 0.75)
                    if fall_speed < MIN_FALL_SPEED:
                        fall_speed = MIN_FALL_SPEED

                # Get the next shape
                current_shape_name, (current_shape, current_color) = next_shape_name, (next_shape, next_color)
                next_shape_name, (next_shape, next_color) = random.choice(SHAPES_LIST)
                shape_pos = [BOARD_WIDTH // 2 - len(current_shape[0]) // 2, 0]
                hold_used = False

                # Reset soft drop time
                soft_drop_time = current_time

                # Check for game over condition
                if not valid_position(board, current_shape, shape_pos):
                    game_over = True

        # Drawing code
        screen.fill(BLACK)
        pygame.draw.line(screen, WHITE, (SCREEN_WIDTH, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), 2)
        draw_board(screen, board)
        draw_shape(screen, current_shape, current_color, shape_pos)
        draw_side_panel(screen, score, level, (next_shape, next_color), held_shape)
        pygame.display.update()

    pygame.quit()






if __name__ == "__main__":

    main()