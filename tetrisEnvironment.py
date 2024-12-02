# tetris_env.py

import numpy as np
import random
import pygame

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

# Define Tetromino shapes with colors
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

# Helper functions
def create_board():
    return [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]

def valid_position(board, shape, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                new_x = x + off_x
                new_y = y + off_y
                if (new_x < 0 or new_x >= BOARD_WIDTH or
                    new_y >= BOARD_HEIGHT or
                    (new_y >= 0 and board[new_y][new_x])):
                    return False
    return True

def add_shape_to_board(board, shape, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, value in enumerate(row):
            if value and y + off_y >= 0:
                board[y + off_y][x + off_x] = 1  # Use 1 to represent filled cell

def clear_lines(board):
    new_board = [row for row in board if any(cell == 0 for cell in row)]
    lines_cleared = BOARD_HEIGHT - len(new_board)
    new_board = [[0] * BOARD_WIDTH for _ in range(lines_cleared)] + new_board
    return new_board, lines_cleared

def rotate_shape(shape):
    return [list(row) for row in zip(*shape[::-1])]

class TetrisEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Tetris AI')
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

        self.action_space = [0, 1, 2, 3, 4]  # Actions: 0-left, 1-right, 2-rotate, 3-soft drop, 4-hard drop
        self.reset()

    def reset(self):
        self.board = create_board()
        self.score = 0
        self.total_lines_cleared = 0
        self.level = 0
        self.fall_speed = 500
        self.game_over = False

        self.current_shape_name, (self.current_shape, _) = random.choice(SHAPES_LIST)
        self.next_shape_name, (self.next_shape, _) = random.choice(SHAPES_LIST)
        self.shape_pos = [BOARD_WIDTH // 2 - len(self.current_shape[0]) // 2, -2]  # Start above the board
        self.hold_used = False
        self.held_shape = None

        self.last_total_lines_cleared = 0

        state = self.get_state()
        return state

    def step(self, action):
        reward = 0
        done = False

        # Apply the action
        self.apply_action(action)

        # Simulate one time step (piece falls by one if possible)
        self.shape_pos[1] += 1
        if not valid_position(self.board, self.current_shape, self.shape_pos):
            self.shape_pos[1] -= 1
            self.lock_piece()
            lines_cleared = self.total_lines_cleared - self.last_total_lines_cleared
            reward += lines_cleared * 10  # Reward for clearing lines
            self.last_total_lines_cleared = self.total_lines_cleared
        else:
            lines_cleared = 0

        # Check if game is over
        if self.game_over:
            done = True
            reward -= 20  # Penalty for game over

        # Get the next state
        state = self.get_state()

        # Additional reward shaping (optional)
        reward -= self.count_holes() * 0.5  # Penalty for holes
        reward -= self.get_bumpiness() * 0.2  # Penalty for uneven surface

        if self.render_mode:
            self.render()

        return state, reward, done, {}

    def apply_action(self, action):
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
            # Soft drop
            self.shape_pos[1] += 1
            if not valid_position(self.board, self.current_shape, self.shape_pos):
                self.shape_pos[1] -= 1
                self.lock_piece()
        elif action == 4:
            # Hard drop
            while valid_position(self.board, self.current_shape, (self.shape_pos[0], self.shape_pos[1] + 1)):
                self.shape_pos[1] += 1
            self.lock_piece()

    def lock_piece(self):
        add_shape_to_board(self.board, self.current_shape, self.shape_pos)
        self.board, lines_cleared = clear_lines(self.board)
        self.total_lines_cleared += lines_cleared
        self.score += lines_cleared * 100

        # Update level and fall speed
        new_level = self.total_lines_cleared // 10
        if new_level > self.level:
            self.level = new_level
            self.fall_speed = max(50, int(self.fall_speed * 0.75))

        # Get next shape
        self.current_shape_name, (self.current_shape, _) = self.next_shape_name, (self.next_shape, _)
        self.next_shape_name, (self.next_shape, _) = random.choice(SHAPES_LIST)
        self.shape_pos = [BOARD_WIDTH // 2 - len(self.current_shape[0]) // 2, -2]

        if not valid_position(self.board, self.current_shape, self.shape_pos):
            self.game_over = True

    def get_state(self):
        # Feature-based state representation
        # Heights of each column
        heights = self.get_column_heights()
        # Differences between adjacent columns
        diffs = [heights[i] - heights[i + 1] for i in range(len(heights) - 1)]
        # Number of holes
        holes = self.count_holes()
        # Bumpiness
        bumpiness = self.get_bumpiness()
        # Aggregate height
        aggregate_height = sum(heights)
        # Max height
        max_height = max(heights)
        # Create feature vector
        state = np.array(heights + [holes, bumpiness, aggregate_height, max_height])
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
            block_found = False
            for y in range(BOARD_HEIGHT):
                if self.board[y][x]:
                    block_found = True
                elif block_found and not self.board[y][x]:
                    holes += 1
        return holes

    def get_bumpiness(self):
        heights = self.get_column_heights()
        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
        return bumpiness

    def render(self):
        if not self.render_mode:
            return
        self.screen.fill(BLACK)
        # Draw board
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x]:
                    pygame.draw.rect(self.screen, GRAY,
                                     (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Draw current shape
        for y, row in enumerate(self.current_shape):
            for x, cell in enumerate(row):
                if cell and y + self.shape_pos[1] >= 0:
                    pygame.draw.rect(self.screen, WHITE,
                                     ((x + self.shape_pos[0]) * BLOCK_SIZE,
                                      (y + self.shape_pos[1]) * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS

    def close(self):
        if self.render_mode:
            pygame.quit()
