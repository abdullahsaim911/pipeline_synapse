"""Synthetic Khan-style test video: black canvas, bright pen/cursor dot.
 0-4 idle | 4-9 WRITING (ink persists) | 9-11 idle | 11-15 cursor moves over ink, NO new ink
 15-17 idle | 17-19 scene cut to a gray slide | 19-21 cut back to black canvas"""
import math
import cv2
import numpy as np

W, H, FPS = 640, 360, 30


def make_video(path: str):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    ink = np.zeros((H, W, 3), np.uint8)
    pos = (100, 180)
    def draw(frame, p):
        cv2.circle(frame, p, 5, (0, 255, 255), -1)
    n = 22 * FPS
    for i in range(n):
        t = i / FPS
        if 4 <= t < 9:
            u = (t - 4) / 5
            new = (int(100 + 440 * u), int(180 + 60 * math.sin(u * 6 * math.pi)))
            cv2.line(ink, pos, new, (255, 255, 255), 2)
            pos = new
        if 11 <= t < 15:
            u = (t - 11) / 4
            pos = (int(540 - 400 * u), int(120 + 80 * math.sin(u * 3 * math.pi)))
        if 17 <= t < 19:
            fr = np.full((H, W, 3), 128, np.uint8)
            cv2.putText(fr, "SLIDE", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            vw.write(fr); continue
        fr = ink.copy()
        draw(fr, pos)
        vw.write(fr)
    vw.release()


def make_video_dark(path: str, W=1280, H=720, ink_bgr=(180, 50, 0), thick=2):
    """Regression video: thin DARK-BLUE ink + noise at 720p (gray thresholds used to miss this)."""
    rng = np.random.default_rng(0)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    ink = np.zeros((H, W, 3), np.uint8); pos = (200, 360)
    for i in range(20 * FPS):
        t = i / FPS
        if 4 <= t < 9:
            u = (t - 4) / 5
            new = (int(200 + 880 * u), int(360 + 120 * math.sin(u * 6 * math.pi)))
            cv2.line(ink, pos, new, ink_bgr, thick, cv2.LINE_AA); pos = new
        if 11 <= t < 15:
            u = (t - 11) / 4
            pos = (int(1080 - 800 * u), int(240 + 160 * math.sin(u * 3 * math.pi)))
        fr = ink.copy(); cv2.circle(fr, pos, 8, (0, 200, 255), -1)
        fr = np.clip(fr.astype(np.float32) + rng.normal(0, 2, fr.shape), 0, 255).astype(np.uint8)
        vw.write(fr)
    vw.release()


def make_pages_video(path: str, pages: int = 2, lines: int = 4, pause: float = 5.0, W: int = 640, H: int = 360):
    """Khan-style: each page is filled line by line (pen + visible pointer), then cleared. Pauses = teacher talking."""
    rng = np.random.default_rng(3)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    colors = [(200, 80, 0), (0, 170, 0), (255, 255, 255)]
    for pg in range(pages):
        ink = np.zeros((H, W, 3), np.uint8); pos = (40, 50)
        for ln in range(lines):
            y0 = 45 + 45 * ln; length = int(rng.integers(200, 450)); col = colors[ln % 3]
            for f in range(int(3.5 * FPS)):
                u = f / (3.5 * FPS)
                new = (int(40 + length * u), int(y0 + 12 * math.sin(u * 14 * math.pi)))
                cv2.line(ink, pos, new, col, 2, cv2.LINE_AA); pos = new
                fr = ink.copy(); cv2.circle(fr, pos, 5, (0, 200, 255), -1); vw.write(fr)
            for f in range(int(pause * FPS)):
                fr = ink.copy(); cv2.circle(fr, pos, 5, (0, 200, 255), -1); vw.write(fr)
            pos = (40, y0 + 45)
        for f in range(int(1.5 * FPS)):
            vw.write(np.zeros((H, W, 3), np.uint8))
    vw.release()
