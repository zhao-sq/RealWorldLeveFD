import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import pyspacemouse
import time
import numpy as np
import cv2
from collections import deque
import threading

# === Parameters ===
BUFFER_SIZE = 600
PLOT_WIDTH = 1000
PLOT_HEIGHT = 400
SCALE = 200.0
FPS = 60
TIME_PER_POINT = 1.0 / FPS

# === Buffers ===
buffer_x = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)
buffer_y = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)
buffer_z = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)

# === Shared state ===
last_state = None
lock = threading.Lock()

# === Callback ===
def handle_spacemouse_input(state):
    global last_state
    with lock:
        last_state = state

# === Plot drawing ===
def draw_spacemouse_plot():
    img = np.ones((PLOT_HEIGHT, PLOT_WIDTH, 3), dtype=np.uint8) * 255
    center_y = PLOT_HEIGHT // 2

    for buffer, color in zip([buffer_x, buffer_y, buffer_z], [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        pts = np.array([
            [int(i * PLOT_WIDTH / BUFFER_SIZE),
             int(center_y - val * SCALE)]
            for i, val in enumerate(buffer)
        ])
        pts = np.clip(pts, [0, 0], [PLOT_WIDTH-1, PLOT_HEIGHT-1])
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2)

    for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        y = int(center_y - val * SCALE)
        cv2.line(img, (0, y), (PLOT_WIDTH, y), (220, 220, 220), 1)
        label = f"{val:+.1f}"
        cv2.putText(img, label, (5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 50), 1)

    ticks_every = int(FPS)
    for i in range(0, BUFFER_SIZE, ticks_every):
        x = int(i * PLOT_WIDTH / BUFFER_SIZE)
        t = (BUFFER_SIZE - i) * TIME_PER_POINT
        cv2.line(img, (x, center_y - 5), (x, center_y + 5), (0, 0, 0), 1)
        cv2.putText(img, f"-{int(t)}s", (x + 2, center_y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    cv2.putText(img, "x=blue, y=green, z=red", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    return img

# === Background thread that continuously calls read() to trigger callbacks ===
def polling_loop():
    while True:
        try:
            pyspacemouse.read()
        except Exception:
            break

# === Start ===
pyspacemouse.open(callback=handle_spacemouse_input)
threading.Thread(target=polling_loop, daemon=True).start()
print("✅ SpaceMouse connected. Press ESC to exit.")

# === Main display loop ===
try:
    while True:
        with lock:
            state = last_state

        if state:
            buffer_x.append(state.x)
            buffer_y.append(state.y)
            buffer_z.append(state.z)
        else:
            buffer_x.append(0)
            buffer_y.append(0)
            buffer_z.append(0)

        plot_img = draw_spacemouse_plot()
        cv2.imshow("SpaceMouse Live Plot", plot_img)

        if cv2.waitKey(int(1000 / FPS)) & 0xFF == 27:
            break
        time.sleep(TIME_PER_POINT)

finally:
    print("Exiting...")
    pyspacemouse.close()
    cv2.destroyAllWindows()
