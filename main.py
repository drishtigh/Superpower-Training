"""
main.py

Demo: 2D maze where a ball is controlled by eye movement using your webcam.

Dependencies: mediapipe, opencv-python, pygame

High-level: capture webcam frames, locate eye/iris center via MediaPipe Face Mesh,
map normalized landmark to game window coordinates and move the ball.

This is a compact, approachable demo — not a production-grade gaze tracker.
Calibration, noise filtering, or advanced gaze estimation can be added later.
"""

import math
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import pygame


# --- Config ---
WINDOW_SIZE = (800, 600)
FPS = 30
BALL_RADIUS = 12
BALL_SPEED = 700.0  # pixels per second (max)
SMOOTHING = 0.8  # 0..1, higher = smoother/slower response
# Enable OpenCV debug overlay window showing camera + detected points
SHOW_DEBUG = True


# Simple maze: 0 = empty, 1 = wall, start 'S', goal 'G'
MAZE = [
	"1111111111111111",
	"1S00000000000001",
	"1011110111111101",
	"1000010000000101",
	"1111010111110101",
	"1000010100000101",
	"1011110101111101",
	"1000000001000001",
	"1111111111111G1",
	"1111111111111111",
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


class GazeTracker:
	"""Capture webcam frames and estimate gaze point using MediaPipe Iris/FaceMesh.

	Output: normalized (x,y) in [0,1] relative to camera frame (x to right, y down).
	"""

	def __init__(self, cam_index=0):
		self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
		self.mp_face = mp.solutions.face_mesh
		self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
												refine_landmarks=True,
												min_detection_confidence=0.5, min_tracking_confidence=0.5)
		# buffer to smooth gaze
		self.buffer = deque(maxlen=5)
		# debug info
		self.last_frame = None
		self.last_left = None
		self.last_right = None
		self.last_gaze = None

	def read_normalized_point(self):
		if not self.cap.isOpened():
			return None
		ret, frame = self.cap.read()
		if not ret:
			return None
		h, w = frame.shape[:2]
		# convert BGR to RGB
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.face_mesh.process(rgb)
		if not results.multi_face_landmarks:
			# store last frame for debug overlay
			if SHOW_DEBUG:
				self.last_frame = frame
				self.last_left = None
				self.last_right = None
				self.last_gaze = None
			return None
		lm = results.multi_face_landmarks[0]
		# Use iris center: MediaPipe with refine_landmarks provides iris landmarks
		# Left iris: landmarks 468..473, Right iris: 473..478
		# We'll average left and right iris centers to get gaze point
		try:
			left_center = self._average_landmark(lm, [468, 469, 470, 471, 472])
			right_center = self._average_landmark(lm, [473, 474, 475, 476, 477])
		except Exception:
			# fallback: use eye corner points if iris not present
			left_center = self._landmark_to_xy(lm, 33)  # left eye outer
			right_center = self._landmark_to_xy(lm, 263)  # right eye outer

		# average both eyes
		nx = (left_center[0] + right_center[0]) / 2.0
		ny = (left_center[1] + right_center[1]) / 2.0
		# append to smoothing buffer
		self.buffer.append((nx, ny))
		avgx = sum(p[0] for p in self.buffer) / len(self.buffer)
		avgy = sum(p[1] for p in self.buffer) / len(self.buffer)
		# normalized coords
		nx = clamp(avgx, 0.0, 1.0)
		ny = clamp(avgy, 0.0, 1.0)
		if SHOW_DEBUG:
			# convert normalized to pixel coords in camera frame for overlay
			h, w = frame.shape[:2]
			self.last_frame = frame
			try:
				lcx = int(left_center[0] * w)
				lcy = int(left_center[1] * h)
				rcx = int(right_center[0] * w)
				rcy = int(right_center[1] * h)
				self.last_left = (lcx, lcy)
				self.last_right = (rcx, rcy)
			except Exception:
				self.last_left = None
				self.last_right = None
			self.last_gaze = (int(nx * w), int(ny * h))
		return nx, ny

	def _average_landmark(self, lm, idxs):
		xs = []
		ys = []
		for i in idxs:
			lmpt = lm.landmark[i]
			xs.append(lmpt.x)
			ys.append(lmpt.y)
		return (sum(xs) / len(xs), sum(ys) / len(ys))

	def _landmark_to_xy(self, lm, idx):
		pt = lm.landmark[idx]
		return pt.x, pt.y

	def release(self):
		try:
			self.cap.release()
		except Exception:
			pass


def collide_rects(circle_pos, radius, walls):
	cx, cy = circle_pos
	for r in walls:
		# find nearest point on rect to circle
		nearest_x = clamp(cx, r.left, r.right)
		nearest_y = clamp(cy, r.top, r.bottom)
		dx = cx - nearest_x
		dy = cy - nearest_y
		if dx * dx + dy * dy < radius * radius:
			return True
	return False


def main():
	pygame.init()
	screen = pygame.display.set_mode(WINDOW_SIZE)
	pygame.display.set_caption("Eye-controlled Maze (demo)")
	clock = pygame.time.Clock()

	walls, start, goal = build_walls(MAZE, WINDOW_SIZE)
	if start is None:
		start_pos = (50, 50)
	else:
		start_pos = start
	if goal is None:
		goal_pos = (WINDOW_SIZE[0] - 50, WINDOW_SIZE[1] - 50)
	else:
		goal_pos = goal

	ball_x, ball_y = start_pos
	gaze = GazeTracker(0)

	last_time = time.time()
	running = True
	paused = False
	status_message = "Looking for face..."

	try:
		while running:
			dt = clock.tick(FPS) / 1000.0
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						running = False
					elif event.key == pygame.K_SPACE:
						paused = not paused

			if not paused:
				gp = gaze.read_normalized_point()
				if gp is None:
					status_message = "No face detected"
				else:
					status_message = "Tracking"
					# Map normalized gaze to window coordinates (x to right, y down)
					gx = gp[0] * WINDOW_SIZE[0]
					gy = gp[1] * WINDOW_SIZE[1]
					# smooth interpolation
					ball_x = SMOOTHING * ball_x + (1 - SMOOTHING) * gx
					ball_y = SMOOTHING * ball_y + (1 - SMOOTHING) * gy
					# limit speed (avoid teleport on large jumps)
					# compute vector to target (gaze point) and move ball max BALL_SPEED*dt
					# but since we already lerped, we just clamp distance move per frame
					# and check collisions

				# collision handling: simple push back if inside wall
				if collide_rects((ball_x, ball_y), BALL_RADIUS, walls):
					# naive simple correction: move ball back towards start a bit
					dx = (start_pos[0] - ball_x) * 0.1
					dy = (start_pos[1] - ball_y) * 0.1
					ball_x += dx
					ball_y += dy

			# Drawing
			screen.fill((20, 20, 30))
			# draw walls
			for r in walls:
				pygame.draw.rect(screen, (200, 200, 200), r)
			# draw goal
			pygame.draw.circle(screen, (40, 200, 40), (int(goal_pos[0]), int(goal_pos[1])), 18)
			# draw ball
			pygame.draw.circle(screen, (200, 40, 40), (int(ball_x), int(ball_y)), BALL_RADIUS)
			# UI text
			font = pygame.font.SysFont(None, 24)
			txt = font.render(f"Status: {status_message}  —  Press SPACE to pause, ESC to quit", True, (220, 220, 220))
			screen.blit(txt, (10, WINDOW_SIZE[1] - 30))

			pygame.display.flip()

			# debug overlay: show camera frame with detected points
			if SHOW_DEBUG:
				frame = gaze.last_frame
				if frame is not None:
					vis = frame.copy()
					# draw left/right iris and gaze
					if gaze.last_left is not None:
						cv2.circle(vis, gaze.last_left, 3, (0, 255, 0), -1)
					if gaze.last_right is not None:
						cv2.circle(vis, gaze.last_right, 3, (0, 255, 0), -1)
					if gaze.last_gaze is not None:
						cv2.circle(vis, gaze.last_gaze, 5, (0, 0, 255), -1)
					# show small scaled window
					try:
						cv2.imshow('Debug - camera', vis)
						# non-blocking key check for OpenCV windows
						if cv2.waitKey(1) & 0xFF == ord('q'):
							running = False
					except Exception:
						# some environments disallow GUI windows; ignore
						pass

			# check win
			if math.hypot(ball_x - goal_pos[0], ball_y - goal_pos[1]) < BALL_RADIUS + 16:
				# display win message and stop
				screen.fill((0, 0, 0))
				big = pygame.font.SysFont(None, 64)
				msg = big.render("You reached the goal!", True, (255, 220, 80))
				screen.blit(msg, (WINDOW_SIZE[0] // 2 - msg.get_width() // 2, WINDOW_SIZE[1] // 2 - 32))
				pygame.display.flip()
				pygame.time.wait(2000)
				running = False

	finally:
		gaze.release()
		try:
			cv2.destroyAllWindows()
		except Exception:
			pass
		pygame.quit()


if __name__ == "__main__":
	main()
