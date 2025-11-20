"""
main.py

Simple Pygame maze where you move a ball with keys:
  W = up, A = left, S = right, Z = down

This is a clean keyboard-only version (no webcam / mediapipe).
"""

import math
import sys
import pygame


# --- Config ---
WINDOW_SIZE = (800, 600)
FPS = 60
BALL_RADIUS = 12
BALL_SPEED = 220.0  # pixels per second


# Maze map: '1' wall, '0' empty, 'S' start, 'G' goal
MAZE = [
	"111111111111",
	"1S0000010001",
	"101110111010",
	"100010000010",
	"111010111010",
	"100000100010",
	"101111101110",
	"1000000000G1",
	"111111111111",
]


def build_walls(maze, window_size):
	rows = len(maze)
	cols = len(maze[0])
	cell_w = window_size[0] / cols
	cell_h = window_size[1] / rows
	walls = []
	start = None
	goal = None
	for r, row in enumerate(maze):
		for c, ch in enumerate(row):
			rect = pygame.Rect(int(c * cell_w), int(r * cell_h), int(cell_w), int(cell_h))
			if ch == "1":
				walls.append(rect)
			elif ch == "S":
				start = rect.center
			elif ch == "G":
				goal = rect.center
	return walls, start, goal


def clamp(v, a, b):
	return max(a, min(b, v))


def circle_rect_collision(cx, cy, radius, rect):
	# nearest point on rect to circle center
	nearest_x = clamp(cx, rect.left, rect.right)
	nearest_y = clamp(cy, rect.top, rect.bottom)
	dx = cx - nearest_x
	dy = cy - nearest_y
	return dx * dx + dy * dy < radius * radius


def collides_any(cx, cy, radius, walls):
	for r in walls:
		if circle_rect_collision(cx, cy, radius, r):
			return True
	return False


def main():
	pygame.init()
	screen = pygame.display.set_mode(WINDOW_SIZE)
	pygame.display.set_caption("Keyboard Maze — W/A/S/Z to move")
	clock = pygame.time.Clock()

	walls, start, goal = build_walls(MAZE, WINDOW_SIZE)
	if start is None:
		ball_x, ball_y = WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2
	else:
		ball_x, ball_y = start
	if goal is None:
		goal_pos = (WINDOW_SIZE[0] - 40, WINDOW_SIZE[1] - 40)
	else:
		goal_pos = goal

	running = True

	font = pygame.font.SysFont(None, 22)

	while running:
		dt = clock.tick(FPS) / 1000.0
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False

		# Movement: W=up, A=left, S=right, Z=down
		keys = pygame.key.get_pressed()
		dx = 0
		dy = 0
		if keys[pygame.K_a]:
			dx -= 1
		if keys[pygame.K_s]:
			dx += 1
		if keys[pygame.K_w]:
			dy -= 1
		if keys[pygame.K_z]:
			dy += 1

		# normalize diagonal movement
		if dx != 0 or dy != 0:
			length = math.hypot(dx, dy)
			dx = dx / length
			dy = dy / length
			move_x = dx * BALL_SPEED * dt
			move_y = dy * BALL_SPEED * dt

			# attempt horizontal move and check collisions (separately for sliding)
			new_x = ball_x + move_x
			if not collides_any(new_x, ball_y, BALL_RADIUS, walls):
				ball_x = new_x
			# attempt vertical move
			new_y = ball_y + move_y
			if not collides_any(ball_x, new_y, BALL_RADIUS, walls):
				ball_y = new_y

		# Drawing
		screen.fill((30, 30, 40))
		for r in walls:
			pygame.draw.rect(screen, (200, 200, 200), r)

		# draw goal
		pygame.draw.circle(screen, (40, 200, 40), (int(goal_pos[0]), int(goal_pos[1])), BALL_RADIUS + 6)

		# draw ball
		pygame.draw.circle(screen, (200, 40, 40), (int(ball_x), int(ball_y)), BALL_RADIUS)

		# HUD
		txt = font.render("Move: W=Up  A=Left  S=Right  Z=Down  —  ESC to quit", True, (220, 220, 220))
		screen.blit(txt, (10, WINDOW_SIZE[1] - 28))

		pygame.display.flip()

		# check win
		if math.hypot(ball_x - goal_pos[0], ball_y - goal_pos[1]) < BALL_RADIUS + 6:
			screen.fill((0, 0, 0))
			big = pygame.font.SysFont(None, 64)
			msg = big.render("You reached the goal!", True, (255, 220, 80))
			screen.blit(msg, (WINDOW_SIZE[0] // 2 - msg.get_width() // 2, WINDOW_SIZE[1] // 2 - 32))
			pygame.display.flip()
			pygame.time.wait(1800)
			running = False

	pygame.quit()


if __name__ == "__main__":
	main()
