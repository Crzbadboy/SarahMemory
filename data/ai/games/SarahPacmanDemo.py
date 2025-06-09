# SarahPacmanDemo.py
# Lightweight Pac-Man clone Sarah can play and learn from

import pygame
import random

pygame.init()
WIDTH, HEIGHT = 640, 480
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sarah's Pac-Man Demo")

FPS = 10
TILE_SIZE = 40
PLAYER_COLOR = (255, 255, 0)
FOOD_COLOR = (255, 0, 0)
BG_COLOR = (0, 0, 0)

player_pos = [WIDTH // 2, HEIGHT // 2]
foods = []
for _ in range(10):
    foods.append([random.randint(0, WIDTH // TILE_SIZE - 1) * TILE_SIZE,
                  random.randint(0, HEIGHT // TILE_SIZE - 1) * TILE_SIZE])


def draw_window():
    WIN.fill(BG_COLOR)
    pygame.draw.rect(WIN, PLAYER_COLOR, (*player_pos, TILE_SIZE, TILE_SIZE))
    for food in foods:
        pygame.draw.rect(WIN, FOOD_COLOR, (*food, TILE_SIZE, TILE_SIZE))
    pygame.display.update()


def move_player(direction):
    if direction == 'left':
        player_pos[0] -= TILE_SIZE
    elif direction == 'right':
        player_pos[0] += TILE_SIZE
    elif direction == 'up':
        player_pos[1] -= TILE_SIZE
    elif direction == 'down':
        player_pos[1] += TILE_SIZE

    player_pos[0] %= WIDTH
    player_pos[1] %= HEIGHT


def check_collisions():
    global foods
    new_foods = []
    for food in foods:
        if player_pos != food:
            new_foods.append(food)
    foods = new_foods


def main():
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            move_player('left')
        if keys[pygame.K_RIGHT]:
            move_player('right')
        if keys[pygame.K_UP]:
            move_player('up')
        if keys[pygame.K_DOWN]:
            move_player('down')

        check_collisions()
        draw_window()

    pygame.quit()


if __name__ == '__main__':
    main()
