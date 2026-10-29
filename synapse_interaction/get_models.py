"""Download the MediaPipe hand model used by HAND_POINTING:  python -m synapse_interaction.get_models

Only needed with current mediapipe (the Tasks API). Run it once; the file is saved inside this package, so it is
found no matter which directory the server starts from. Nothing is downloaded unless you run this."""
import os
import urllib.request

from .config import Config

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def main() -> None:
    dest = Config().hand_model_path
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"already present: {dest}")
        return
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, dest)
    print(f"saved {dest} ({os.path.getsize(dest) // 1024} KB)")


if __name__ == "__main__":
    main()
