# manual_test.py

from tetrisEnvironment import TetrisEnv  # Import the tetrisEnvironment class
import pygame

def manual_test():
    env = TetrisEnv(render_mode=True)
    state = env.reset()
    done = False

    clock = pygame.time.Clock()

    # Initialize key timers
    key_timers = {
        'left': 0,
        'right': 0,
        'rotate': 0,
        'hard_drop': 0,
        'hold': 0
    }

    # Set key repeat delay and interval in milliseconds
    key_repeat_delay = 150  # Initial delay before repeating
    key_repeat_interval = 100  # Interval between repeats

    while not done:
        # Limit the frame rate to 60 FPS
        clock.tick(60)
        current_time = pygame.time.get_ticks()
        action = None

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                env.close()
                pygame.quit()
                return

        # Get the state of all keys
        keys = pygame.key.get_pressed()

        # Movement flags
        move_left = keys[pygame.K_LEFT]
        move_right = keys[pygame.K_RIGHT]
        soft_drop = keys[pygame.K_DOWN]
        rotate = keys[pygame.K_UP]
        hard_drop = keys[pygame.K_SPACE]
        hold = keys[pygame.K_c]

        # Initialize action to None
        action = None

        # Handle soft drop action separately
        if soft_drop:
            action = 3  # Soft drop
        else:
            # Check for movement
            if move_left:
                if current_time - key_timers['left'] > key_repeat_interval:
                    action = 0  # Move left
                    key_timers['left'] = current_time
            if move_right:
                if current_time - key_timers['right'] > key_repeat_interval:
                    action = 1  # Move right
                    key_timers['right'] = current_time
            if rotate:
                if current_time - key_timers['rotate'] > key_repeat_delay:
                    action = 2  # Rotate
                    key_timers['rotate'] = current_time
            if hard_drop:
                if current_time - key_timers['hard_drop'] > key_repeat_delay:
                    action = 4  # Hard drop
                    key_timers['hard_drop'] = current_time
            if hold:
                if current_time - key_timers['hold'] > key_repeat_delay:
                    action = 5  # Holdgim
                    key_timers['hold'] = current_time

        # Apply action if any
        if action is not None:
            state, reward, done, _ = env.step(action)
        else:
            # Let the piece fall naturally based on the gravity
            state, reward, done, _ = env.step(action=None)

        env.render()

    env.close()

if __name__ == "__main__":
    manual_test()
