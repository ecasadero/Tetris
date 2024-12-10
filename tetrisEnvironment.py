# tetrisEnvironment.py

import numpy as np
import random
import pygame
import sqlite3

# Initialize Pygame
pygame.init()

# Screen size and block settings
BLOCK_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
SCREEN_WIDTH = BOARD_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = BOARD_HEIGHT * BLOCK_SIZE
SIDE_PANEL_WIDTH = 200  # For side panel

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

# Defines tetromino shapes with colors
SHAPES_WITH_COLORS = {
    "IShape": ([[1, 1, 1, 1]], CYAN),
    "OShape": ([[1, 1], [1, 1]], YELLOW),
    "TShape": ([[0, 1, 0], [1, 1, 1]], MAGENTA),
    "SShape": ([[0, 1, 1], [1, 1, 0]], GREEN),
    "ZShape": ([[1, 1, 0], [0, 1, 1]], RED),
    "JShape": ([[1, 0, 0], [1, 1, 1]], BLUE),
    "LShape": ([[0, 0, 1], [1, 1, 1]], ORANGE),
}

SHAPES_LIST = list(SHAPES_WITH_COLORS.items())

# Initialize the database
def initialize_database():
    conn = sqlite3.connect('tetris_stats.db')
    c = conn.cursor()

    # Create the runs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP
        )
    ''')

    # Create the games table
    c.execute('''
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            highest_level INTEGER DEFAULT 0,
            total_score INTEGER DEFAULT 0,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
    ''')

    # Create the game_stats table
    c.execute('''
        CREATE TABLE IF NOT EXISTS game_stats (
            game_id INTEGER,
            piece_type TEXT,
            occurrences INTEGER,
            single_lines INTEGER,
            double_lines INTEGER,
            triple_lines INTEGER,
            tetris_lines INTEGER,
            total_lines INTEGER,
            PRIMARY KEY (game_id, piece_type),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    ''')
    # Create the episode_rewards table
    c.execute('''
            CREATE TABLE IF NOT EXISTS episode_rewards (
                run_id INTEGER,
                episode INTEGER,
                total_reward REAL,
                pieces_placed INTEGER DEFAULT 0,
                PRIMARY KEY (run_id, episode),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        ''')
    conn.commit()
    conn.close()


initialize_database()
def update_database_schema():
    conn = sqlite3.connect('tetris_stats.db')
    c = conn.cursor()

    # Add the `pieces_placed` column if it doesn't already exist
    try:
        c.execute("ALTER TABLE episode_rewards ADD COLUMN pieces_placed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        # Column already exists
        pass

    conn.commit()
    conn.close()

# Call this function once during initialization
update_database_schema()

def start_new_run():
    conn = sqlite3.connect('tetris_stats.db')
    c = conn.cursor()

    # Insert a new run record
    c.execute('INSERT INTO runs DEFAULT VALUES')
    run_id = c.lastrowid  # Get the new run_id

    conn.commit()
    conn.close()

    return run_id


def start_new_game(run_id):
    conn = sqlite3.connect('tetris_stats.db')
    c = conn.cursor()

    # Insert a new game record
    c.execute('INSERT INTO games (run_id) VALUES (?)', (run_id,))
    game_id = c.lastrowid  # Get the new game_id

    conn.commit()
    conn.close()

    return game_id


def end_run(run_id):
    conn = sqlite3.connect('tetris_stats.db')
    c = conn.cursor()

    c.execute('''
        UPDATE runs
        SET end_time = CURRENT_TIMESTAMP
        WHERE run_id = ?
    ''', (run_id,))

    conn.commit()
    conn.close()

def initialize_game_stats(game_id):
    conn = sqlite3.connect('tetris_stats.db')
    c = conn.cursor()

    tetrominoes = ['IShape', 'OShape', 'TShape', 'SShape', 'ZShape', 'JShape', 'LShape']
    for piece in tetrominoes:
        c.execute('''
            INSERT OR IGNORE INTO game_stats (
                game_id, piece_type, occurrences, single_lines, double_lines, triple_lines, tetris_lines, total_lines
            ) VALUES (?, ?, 0, 0, 0, 0, 0, 0)
        ''', (game_id, piece))

    conn.commit()
    conn.close()

def record_tetromino(game_id, piece_type):
    conn = sqlite3.connect('tetris_stats.db')
    c = conn.cursor()
    c.execute('''
        UPDATE game_stats
        SET occurrences = occurrences + 1
        WHERE game_id = ? AND piece_type = ?
    ''', (game_id, piece_type))
    conn.commit()
    conn.close()

def save_episode_reward(run_id, episode, total_reward, pieces_placed):
        conn = sqlite3.connect('tetris_stats.db')
        c = conn.cursor()

        c.execute('''
            INSERT INTO episode_rewards (run_id, episode, total_reward, pieces_placed)
            VALUES (?, ?, ?,?)
        ''', (run_id, episode, total_reward, pieces_placed))

        conn.commit()
        conn.close()


def record_line_clears(game_id, lines_cleared, piece_type):
    conn = sqlite3.connect('tetris_stats.db')
    c = conn.cursor()

    # Update total lines cleared
    c.execute('''
        UPDATE game_stats
        SET total_lines = total_lines + ?
        WHERE game_id = ? AND piece_type = ?
    ''', (lines_cleared, game_id, piece_type))

    # Update specific line clear counts
    if lines_cleared == 1:
        c.execute('UPDATE game_stats SET single_lines = single_lines + 1 WHERE game_id = ? AND piece_type = ?',
                  (game_id, piece_type))
    elif lines_cleared == 2:
        c.execute('UPDATE game_stats SET double_lines = double_lines + 1 WHERE game_id = ? AND piece_type = ?',
                  (game_id, piece_type))
    elif lines_cleared == 3:
        c.execute('UPDATE game_stats SET triple_lines = triple_lines + 1 WHERE game_id = ? AND piece_type = ?',
                  (game_id, piece_type))
    elif lines_cleared == 4:
        c.execute('UPDATE game_stats SET tetris_lines = tetris_lines + 1 WHERE game_id = ? AND piece_type = ?',
                  (game_id, piece_type))

    conn.commit()
    conn.close()

# Helper functions
def create_board():
    return [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]

def draw_board(screen, board):
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if board[y][x]:
                faded_color = washed_out_color(board[y][x])
                draw_block(screen, faded_color, (x, y))

def draw_shape(screen, shape, color, position):
    for y, row in enumerate(shape):
        for x, value in enumerate(row):
            if value and y + position[1] >= 0:
                draw_block(screen, color, (x + position[0], y + position[1]))

def valid_position(board, shape, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                new_x = x + off_x
                new_y = y + off_y

                # Checks horizontal boundaries
                if new_x < 0 or new_x >= BOARD_WIDTH:
                    return False

                # Checks vertical boundaries
                if new_y >= BOARD_HEIGHT:
                    return False

                # Checks collisions with existing blocks in the board
                if new_y >= 0:
                    if board[new_y][new_x]:
                        return False
                else:
                    # If new_y is negative, and there's a block in the same column at y=0, it's invalid
                    if board[0][new_x]:
                        return False
    return True

def add_shape_to_board(board, shape, color, offset, shape_type, game_id):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, value in enumerate(row):
            if value and y + off_y >= 0:
                board[y + off_y][x + off_x] = color  # Store the block's color

    # Record the placement of the tetromino
    record_tetromino(game_id, shape_type)

def draw_block(screen, color, position):
    x, y = position[0] * BLOCK_SIZE, position[1] * BLOCK_SIZE
    pygame.draw.rect(screen, BLACK, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))
    inner_margin = 2
    pygame.draw.rect(screen, color, pygame.Rect(
        x + inner_margin, y + inner_margin,
        BLOCK_SIZE - 2 * inner_margin, BLOCK_SIZE - 2 * inner_margin
    ))

def washed_out_color(color):
    r, g, b = color
    return ((r + 255) // 2, (g + 255) // 2, (b + 255) // 2)

def clear_lines(board):
    new_board = [row for row in board if any(cell == 0 for cell in row)]
    lines_cleared = BOARD_HEIGHT - len(new_board)
    new_board = [[0] * BOARD_WIDTH for _ in range(lines_cleared)] + new_board
    return new_board, lines_cleared

def rotate_shape(shape):
    return [list(row) for row in zip(*shape[::-1])]

def draw_side_panel(screen, score, level, next_shape_data, held_shape_data):
    font = pygame.font.SysFont("Times New Roman", 24)

    # Draw Score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (SCREEN_WIDTH + 10, SCREEN_HEIGHT // 2 - 80))

    # Draw Level
    level_text = font.render(f"Level: {level}", True, WHITE)
    screen.blit(level_text, (SCREEN_WIDTH + 10, SCREEN_HEIGHT // 2 - 15))

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
    screen.blit(hold_text, (SCREEN_WIDTH + 10, SCREEN_HEIGHT - 195))
    if held_shape_data:
        held_shape, held_color = held_shape_data
        for y, row in enumerate(held_shape):
            for x, value in enumerate(row):
                if value:
                    draw_block(screen, held_color, (x + BOARD_WIDTH + 1, y + BOARD_HEIGHT - 5))

class TetrisEnv:
    def __init__(self, render_mode=False, run_id=None):
        self.render_mode = render_mode
        self.run_id = run_id  # Store the run_id
        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH + SIDE_PANEL_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('CSU Tetris AI')
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

        # Initialize gravity variables before reset
        self.normal_gravity_interval = 500  # Normal gravity interval in milliseconds
        self.gravity_interval = self.normal_gravity_interval  # Current gravity interval
        self.gravity_timer = pygame.time.get_ticks()
        self.soft_drop_active = False  # Flag to indicate if soft drop is active
        self.action_space = [0, 1, 2, 3, 4, 5]  # Actions: 0-left, 1-right, 2-rotate, 3-soft drop, 4-hard drop, 5-hold
        self.reset()

    def reset(self):
        self.board = create_board()
        self.score = 0
        self.total_lines_cleared = 0
        self.level = 0
        self.highest_level = 0
        self.fall_speed = 500
        self.game_over = False

        self.current_shape_name, (self.current_shape, self.current_color) = random.choice(SHAPES_LIST)
        self.next_shape_name, (self.next_shape, self.next_color) = random.choice(SHAPES_LIST)
        self.held_shape = None
        self.hold_used = False
        self.shape_pos = [BOARD_WIDTH // 2 - len(self.current_shape[0]) // 2, -2]

        self.last_total_lines_cleared = 0
        self.piece_locked = False  # Flag to indicate if the piece was locked during apply_action

        # Start a new game session and initialize stats
        self.game_id = start_new_game(self.run_id)
        initialize_game_stats(self.game_id)

        state = self.get_state()
        self.gravity_timer = pygame.time.get_ticks()  # Reset gravity timer
        return state

    def step(self, action):
        current_time = pygame.time.get_ticks()
        reward = 0
        done = False
        self.piece_locked = False  # Reset the flag at the start of each step

        # Apply the action
        self.apply_action(action)

        # Adjust gravity interval based on soft drop
        if self.soft_drop_active:
            self.gravity_interval = self.normal_gravity_interval / 3  # Use float division
        else:
            self.gravity_interval = self.normal_gravity_interval

        # Handle gravity based on time
        if current_time - self.gravity_timer >= self.gravity_interval:
            self.shape_pos[1] += 1
            if not valid_position(self.board, self.current_shape, self.shape_pos):
                self.shape_pos[1] -= 1
                self.lock_piece()
                self.piece_locked = True
            self.gravity_timer = current_time

        # Reward system for locked pieces
        if self.piece_locked:
            lines_cleared = self.total_lines_cleared - self.last_total_lines_cleared
            reward += lines_cleared * 500  # Reward for clearing lines
            if lines_cleared == 4:
                reward += 3000  # Additional reward for Tetris
            self.last_total_lines_cleared = self.total_lines_cleared

            # Additional reward for pieces placed
            reward += 10  # Flat reward for placing a piece

            # Calculate gap reward
            gap_reward = self.calculate_gap_reward(self.current_shape, self.shape_pos)
            reward += gap_reward

        # Check if game is over
        if self.game_over:
            done = True
            reward -= 20  # Penalty for game over

        if not done:
            reward += 5  # Positive reward per time step

        # Get the next state
        state = self.get_state()

        # Additional reward shaping
        holes = self.count_holes()
        bumpiness = self.get_bumpiness()
        std_dev = self.get_standard_deviation()
        edge_util = self.get_edge_utilization()
        coverage = self.get_column_coverage()
        max_height = self.get_max_height()

        # Adjust reward based on new metrics
        reward -= holes * 1  # Penalty for holes
        reward -= bumpiness * 1  # Penalty for uneven surface
        reward -= std_dev * 2  # Penalty for high standard deviation
        reward += edge_util * .01  # Reward for edge utilization
        reward += coverage * 4  # Reward for column coverage
        reward -= max_height * 1  # Penalty for high columns

        if self.render_mode:
            self.render()

        return state, reward, done, {}

    def apply_action(self, action):
        # Reset soft drop flag unless the action is soft drop
        previous_soft_drop = self.soft_drop_active
        self.soft_drop_active = False

        if action == 0:
            # Move left
            self.shape_pos[0] -= 1
            if not valid_position(self.board, self.current_shape, self.shape_pos):
                self.shape_pos[0] += 1
        elif action == 1:
            # Move right
            self.shape_pos[0] += 1
            if not valid_position(self.board, self.current_shape, self.shape_pos):
                self.shape_pos[0] -= 1
        elif action == 2:
            # Rotate
            rotated_shape = rotate_shape(self.current_shape)
            if valid_position(self.board, rotated_shape, self.shape_pos):
                self.current_shape = rotated_shape
        elif action == 3:
            # Soft drop - Increase gravity speed by 3x and move down immediately
            self.soft_drop_active = True
            self.shape_pos[1] += 1
            if not valid_position(self.board, self.current_shape, self.shape_pos):
                self.shape_pos[1] -= 1
                self.lock_piece()
                self.piece_locked = True
                # Reset gravity timer to prevent immediate extra drop
                self.gravity_timer = pygame.time.get_ticks()
            # Debugging Statement (Optional)
            # print(f"Action: Soft Drop {'Activated' if not previous_soft_drop else 'Continued'}")
        elif action == 4:
            # Hard drop
            while valid_position(self.board, self.current_shape, (self.shape_pos[0], self.shape_pos[1] + 1)):
                self.shape_pos[1] += 1
            self.lock_piece()
            self.piece_locked = True
            # Reset gravity timer to prevent immediate extra drop
            self.gravity_timer = pygame.time.get_ticks()
            # Debugging Statement (Optional)
            # print("Action: Hard Drop")
        elif action == 5:
            # Hold
            if not self.hold_used:
                if self.held_shape is None:
                    self.held_shape = (self.current_shape_name, self.current_shape, self.current_color)
                    self._get_next_shape()
                else:
                    # Swap held and current shapes
                    self.held_shape, (self.current_shape_name, self.current_shape, self.current_color) = \
                        (self.current_shape_name, self.current_shape, self.current_color), self.held_shape
                self.shape_pos = [BOARD_WIDTH // 2 - len(self.current_shape[0]) // 2, -2]
                self.hold_used = True
                # Debugging Statement (Optional)
                # print("Action: Hold")

    def _get_next_shape(self):
        self.current_shape_name = self.next_shape_name
        self.current_shape = self.next_shape
        self.current_color = self.next_color
        self.next_shape_name, (self.next_shape, self.next_color) = random.choice(SHAPES_LIST)

    def lock_piece(self):
        off_x, off_y = self.shape_pos

        # Add the shape to the board
        add_shape_to_board(self.board, self.current_shape, self.current_color, self.shape_pos, self.current_shape_name,
                           self.game_id)

        # Reward for filling gaps
        gap_reward = self.calculate_gap_reward(self.current_shape, self.shape_pos)
        self.score += gap_reward

        # Clear lines
        self.board, lines_cleared = clear_lines(self.board)
        self.total_lines_cleared += lines_cleared
        self.score += lines_cleared * 100

        # Record line clears
        if lines_cleared > 0:
            record_line_clears(self.game_id, lines_cleared, self.current_shape_name)

        # Update level and fall speed
        new_level = self.total_lines_cleared // 10
        if new_level > self.level:
            self.level = new_level
            self.fall_speed = max(50, int(self.fall_speed * 0.75))
            self.normal_gravity_interval = max(50, self.normal_gravity_interval * 3 / 4)
            self.gravity_interval = self.normal_gravity_interval

        # Update highest level
        if self.level > self.highest_level:
            self.highest_level = self.level

        # Get the next shape
        self._get_next_shape()
        self.shape_pos = [BOARD_WIDTH // 2 - len(self.current_shape[0]) // 2, -2]
        self.hold_used = False
        self.soft_drop_active = False
        self.gravity_timer = pygame.time.get_ticks()

        # Check for game over
        if not valid_position(self.board, self.current_shape, self.shape_pos):
            print("Game Over: Cannot place new piece.")
            self.game_over = True

    def get_state(self):
        heights = self.get_column_heights()
        holes = self.count_holes()
        bumpiness = self.get_bumpiness()
        aggregate_height = sum(heights)
        max_height = max(heights)
        column_indices = list(range(BOARD_WIDTH))  # [0, 1, ..., 9]

        # Normalize features
        heights = np.array(heights) / BOARD_HEIGHT
        holes = np.array([holes]) / (BOARD_WIDTH * BOARD_HEIGHT)
        bumpiness = np.array([bumpiness]) / (BOARD_WIDTH * BOARD_HEIGHT)
        aggregate_height = np.array([aggregate_height]) / (BOARD_WIDTH * BOARD_HEIGHT)
        max_height = np.array([max_height]) / BOARD_HEIGHT
        column_indices = np.array(column_indices) / BOARD_WIDTH

        state = np.concatenate([heights, holes, bumpiness, aggregate_height, max_height, column_indices])
        return state.astype(np.float32)

    def get_column_heights(self):
        heights = [0] * BOARD_WIDTH
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if self.board[y][x]:
                    heights[x] = BOARD_HEIGHT - y
                    break
        return heights

    def count_holes(self):
        holes = 0
        for x in range(BOARD_WIDTH):
            block_count_above_hole = 0
            for y in range(BOARD_HEIGHT):
                if self.board[y][x]:  # If there's a block
                    block_count_above_hole += 1
                elif block_count_above_hole >= 2 and not self.board[y][
                    x]:  # If it's a hole with at least 2 blocks above
                    holes += 1
        return holes

    def calculate_gap_reward(self, shape, position):
        off_x, off_y = position
        gap_reward = 0

        # Check the rows affected by the piece
        for y, row in enumerate(shape):
            row_index = y + off_y
            if 0 <= row_index < BOARD_HEIGHT:  # Ensure row index is valid
                board_row = self.board[row_index]
                filled_blocks = sum(1 for cell in board_row if cell != 0)
                gaps = sum(1 for cell in board_row if cell == 0 and filled_blocks > 0)

                # Reward for filling gaps
                gap_reward += gaps * filled_blocks * 5 # More filled blocks means higher reward

        return gap_reward

    def get_bumpiness(self):
        heights = self.get_column_heights()
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
        return bumpiness

    # Helper methods for reward shaping
    def get_standard_deviation(self):
        heights = self.get_column_heights()
        return np.std(heights)

    def get_edge_utilization(self):
        edge_columns = [0, BOARD_WIDTH - 1]
        # Count the number of blocks in the edge columns
        edge_height = sum(1 for x in edge_columns for y in range(BOARD_HEIGHT) if self.board[y][x])
        return edge_height

    def get_max_height(self):
        heights = self.get_column_heights()
        return max(heights)

    def get_column_coverage(self):
        coverage = sum([1 for x in range(BOARD_WIDTH) if any(self.board[y][x] for y in range(BOARD_HEIGHT))])
        return coverage

    def render(self):
        if not self.render_mode:
            return
        self.screen.fill(BLACK)

        # Handle Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Draw board
        draw_board(self.screen, self.board)
        # Draw current shape
        draw_shape(self.screen, self.current_shape, self.current_color, self.shape_pos)
        # Draw side panel
        draw_side_panel(self.screen, self.score, self.level, (self.next_shape, self.next_color),
                        (self.held_shape[1], self.held_shape[2]) if self.held_shape else None)
        pygame.display.flip()
        self.clock.tick(60)  # Increase frame rate for smoother rendering

    def close(self):
        if self.render_mode:
            pygame.quit()
            self.render_mode = False  # Ensure we don't try to render after closing

        # Update the games table
        self.end_game_session()

    def end_game_session(self):
        conn = sqlite3.connect('tetris_stats.db')
        c = conn.cursor()
        c.execute('''
            UPDATE games
            SET end_time = CURRENT_TIMESTAMP,
                highest_level = ?,
                total_score = ?
            WHERE game_id = ?
        ''', (self.highest_level, self.score, self.game_id))
        conn.commit()
        conn.close()
