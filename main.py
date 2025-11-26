"""
main.py

Simple Pygame maze where you move a ball with keys:
  W = up, A = left, S = right, Z = down

This is a clean keyboard-only version (no webcam / mediapipe).
"""

import math
import sys
import pygame
import cv2
from collections import deque


# --- Config ---
WINDOW_SIZE = (800, 600)
FPS = 60
BALL_RADIUS = 12
BALL_SPEED = 220.0  # pixels per second
SMOOTHING = 0.85  # gaze smoothing (higher = slower)
SHOW_DEBUG = True  # show camera debug window


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


class GazeTrackerOpenCV:
	"""Simple gaze-ish tracker using OpenCV Haar cascades.

	Output: normalized (x,y) in [0,1] relative to the detected face box (x to right, y down).
	This is an approximate method (pupil detection via dark blob) and works best in decent lighting.
	"""

	def __init__(self, cam_index=0):
		self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
		# haar cascades included with OpenCV
		self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
		self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
		self.buffer = deque(maxlen=6)
		self.last_frame = None

	def read_normalized_point(self):
		if not self.cap.isOpened():
			return None
		ret, frame = self.cap.read()
		if not ret:
			return None
		self.last_frame = frame.copy()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
		if len(faces) == 0:
			return None
		# pick largest face
		faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
		x, y, w, h = faces[0]
		face_roi = gray[y : y + h, x : x + w]
		eyes = self.eye_cascade.detectMultiScale(face_roi)
		if len(eyes) == 0:
			return None
		# compute eye centers in face coordinates
		centers = []
		for (ex, ey, ew, eh) in eyes[:2]:
			# try to refine pupil location by finding the darkest point inside the eye region
			try:
				eye_roi = face_roi[ey : ey + eh, ex : ex + ew]
				if eye_roi.size == 0:
					raise ValueError("empty eye ROI")
				blur = cv2.GaussianBlur(eye_roi, (7, 7), 0)
				(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)
				px, py = minLoc
				# pupil candidate in face coords
				cx = ex + int(px)
				cy = ey + int(py)
			except Exception:
				# fallback to eye bounding-box center
				cx = ex + ew // 2
				cy = ey + eh // 2
			centers.append((cx / w, cy / h))
		if len(centers) == 0:
			return None
		# average eye centers
		ax = sum(c[0] for c in centers) / len(centers)
		ay = sum(c[1] for c in centers) / len(centers)

		# refine using dark blob inside eye region to approximate pupil (optional)
		# map normalized face-relative to global normalized (0..1 across face box)
		nx = ax
		ny = ay
		self.buffer.append((nx, ny))
		avgx = sum(p[0] for p in self.buffer) / len(self.buffer)
		avgy = sum(p[1] for p in self.buffer) / len(self.buffer)
		# store for debug overlay (pixel coords)
		self.last_face = (x, y, w, h)
		self.last_norm = (avgx, avgy)
		return avgx, avgy

	def release(self):
		try:
			self.cap.release()
		except Exception:
			pass


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
	# Gaze tracker (OpenCV) — toggle with 'g'
	gaze = GazeTrackerOpenCV()
	use_gaze = False

	font = pygame.font.SysFont(None, 22)

	while running:
		dt = clock.tick(FPS) / 1000.0
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_g:
					# toggle gaze mode on key press
					use_gaze = not use_gaze

		keys = pygame.key.get_pressed()

		if use_gaze:
			gp = gaze.read_normalized_point()
			if gp is not None:
				# map normalized gaze (relative to face box) to screen coordinates
				fx, fy, fw, fh = getattr(gaze, 'last_face', (0, 0, WINDOW_SIZE[0], WINDOW_SIZE[1]))
				tx = fx + gp[0] * fw
				ty = fy + gp[1] * fh
				# compute a smoothed target position
				target_x = SMOOTHING * ball_x + (1 - SMOOTHING) * tx
				target_y = SMOOTHING * ball_y + (1 - SMOOTHING) * ty
				# move toward target but cap movement by BALL_SPEED * dt and respect collisions
				vx = target_x - ball_x
				vy = target_y - ball_y
				dist = math.hypot(vx, vy)
				if dist > 1e-6:
					maxstep = BALL_SPEED * dt
					step = min(dist, maxstep)
					nx = ball_x + (vx / dist) * step
					ny = ball_y + (vy / dist) * step
					# attempt horizontal move and check collisions (separately for sliding)
					if not collides_any(nx, ball_y, BALL_RADIUS, walls):
						ball_x = nx
					# attempt vertical move
					if not collides_any(ball_x, ny, BALL_RADIUS, walls):
						ball_y = ny
		else:
			# Movement: W=up, A=left, S=right, Z=down
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
		mode = "Gaze" if use_gaze else "Keys"
		txt = font.render(f"Mode: {mode}  —  Move: W=Up  A=Left  S=Right  Z=Down  —  G to toggle gaze  —  ESC to quit", True, (220, 220, 220))
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
	gaze.release()


if __name__ == "__main__":
	main()
