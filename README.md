# Superpower-Training

Eye-controlled 2D Maze Demo

This small demo shows a 2D maze where you move a ball by the movement of your eyes using a standard webcam.

Requirements
- Python 3.8+
- A reasonably recent webcam

Install
Windows PowerShell commands (run in the project folder):

```powershell
python -m pip install -r requirements.txt
```

Run

```powershell
python main.py
```

Controls
- SPACE: pause/unpause
- ESC: quit

Notes
- This demo uses MediaPipe Face Mesh to get iris/eye landmarks from the webcam. It's a simple approach and will need calibration and filtering for production use.
- If MediaPipe cannot access your camera, try running in a different environment or check camera permissions in Windows.
- If performance is slow, reduce `WINDOW_SIZE` or `FPS` in `main.py`.

Next steps (ideas)
- Add a calibration step so gaze maps more accurately to game coordinates.
- Improve collision handling and ball physics.
- Support external eye-trackers (Tobii) or more advanced gaze estimation.