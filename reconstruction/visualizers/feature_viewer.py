import sqlite3

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


class FeatureViewer:
    def __init__(self, database_path, image_dir):
        self.image_dir = Path(image_dir)
        self.conn = sqlite3.connect(database_path)
        self.cursor = self.conn.cursor()

        # Load all image data from DB
        self.cursor.execute("SELECT image_id, name FROM images")
        self.images = self.cursor.fetchall()
        self.current_idx = 0

        if not self.images:
            print("No images found in database.")
            return

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show_image()
        plt.show()

    def show_image(self):
        image_id, name = self.images[self.current_idx]

        # Fetch keypoints
        self.cursor.execute(
            "SELECT data FROM keypoints WHERE image_id = ?", (image_id,)
        )
        row = self.cursor.fetchone()

        if row is None:
            print(f"No keypoints for {name}")
            return

        # SIFT keypoints typically have 6 columns
        keypoints_raw = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 6)

        # Load and draw
        img = cv2.imread(str(self.image_dir / name))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for kp in keypoints_raw:
            cv2.circle(img_rgb, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

        self.ax.clear()
        self.ax.imshow(img_rgb)
        self.ax.set_title(
            f"Image {self.current_idx + 1}/{len(self.images)}: {name} (Press Left/Right)"
        )
        self.ax.axis("off")
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == "right":
            self.current_idx = (self.current_idx + 1) % len(self.images)
        elif event.key == "left":
            self.current_idx = (self.current_idx - 1) % len(self.images)
        else:
            return
        self.show_image()

    def __del__(self):
        if hasattr(self, "conn"):
            self.conn.close()
