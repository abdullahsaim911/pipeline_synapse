"""Find uncovered full-height panels using visible dividers, not guessed tiles."""

import numpy as np


def uncovered_panels(image, evidence):
    """Conservative CoR region proposals; no semantic or OCR claims.

    Only a narrow, high-contrast divider spanning >=85% of the image may
    establish two panels. Existing regions covering a panel suppress the proposal.
    No recursive segmentation or new detector/model dependency is introduced.
    """
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    foreground = gray > 220 if np.median(gray) < 128 else gray < 35
    columns = np.flatnonzero(foreground.mean(axis=0) >= 0.85)
    groups = np.split(columns, np.where(np.diff(columns) > 1)[0] + 1)
    dividers = []
    for group in groups:
        if not len(group) or len(group) > max(4, width // 100):
            continue
        x = int(group[len(group) // 2])
        if not 0.2 * width < x < 0.8 * width:
            continue
        margin = max(4, width // 100)
        left, right = max(0, int(group[0]) - margin), min(width, int(group[-1]) + margin)
        surround = np.concatenate([gray[:, left:int(group[0])].ravel(), gray[:, int(group[-1]) + 1:right].ravel()])
        if not surround.size or abs(float(np.median(gray[:, group])) - float(np.median(surround))) < 80:
            continue
        dividers.append(x)
    if not dividers:
        return []
    x = min(dividers, key=lambda value: abs(value - width / 2)) / width
    candidates = [[0.0, 0.0, x, 1.0], [x, 0.0, 1.0, 1.0]]
    covered_pixels = np.zeros(gray.shape, dtype=bool)
    for region in evidence.get("regions", []):
        box = region.get("bbox_normalized")
        if box:
            covered_pixels[int(box[1] * height):int(box[3] * height),
                           int(box[0] * width):int(box[2] * width)] = True
    result = []
    for box in candidates:
        covered = covered_pixels[:, int(box[0] * width):int(box[2] * width)].mean()
        pixels = gray[:, int(box[0] * width):int(box[2] * width)].astype(float)
        has_content = pixels.size and (np.abs(pixels - np.median(pixels)) > 30).mean() > 0.005
        if covered < 0.6 and has_content:
            result.append(box)
    return result
