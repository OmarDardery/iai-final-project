import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Player
player_width = 50
player_height = 50
player_x = SCREEN_WIDTH // 2 - player_width // 2
player_y = SCREEN_HEIGHT - player_height - 20
player_speed = 5

# Obstacles
obstacle_width = 50
obstacle_height = 50
obstacle_speed = 5
obstacle_gap = 200
obstacle_frequency = 50
obstacles = []

# Score
score = 0
font = pygame.font.SysFont(None, 36)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Subway Surfer")

clock = pygame.time.Clock()

# Game loop
running = True
while running:
    screen.fill(WHITE)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Player movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_x -= player_speed
    if keys[pygame.K_RIGHT]:
        player_x += player_speed
    
    # Spawn obstacles
    if random.randrange(0, 100) < obstacle_frequency:
        obstacle_x = random.randrange(0, SCREEN_WIDTH - obstacle_width)
        obstacle_y = -obstacle_height
        obstacles.append([obstacle_x, obstacle_y])
    
    # Move and draw obstacles
    for obstacle in obstacles:
        obstacle[1] += obstacle_speed
        pygame.draw.rect(screen, BLACK, [obstacle[0], obstacle[1], obstacle_width, obstacle_height])
    
    # Remove obstacles that have gone off screen
    obstacles = [[x, y] for x, y in obstacles if y < SCREEN_HEIGHT]
    
    # Player collision with obstacles
    for obstacle in obstacles:
        if obstacle[1] + obstacle_height > player_y and player_x < obstacle[0] + obstacle_width and player_x + player_width > obstacle[0]:
            # Game over
            running = False
    
    # Draw player
    pygame.draw.rect(screen, BLACK, [player_x, player_y, player_width, player_height])
    
    # Update score
    score += 1
    score_text = font.render("Score: " + str(score), True, BLACK)
    screen.blit(score_text, (10, 10))
    
    pygame.display.update()
    clock.tick(60)

pygame.quit()

