import json
import sys
from pathlib import Path

import cv2
import numpy as np
from attrs import asdict
from PIL import Image

from human_val import Validator

PARENT = Path(__file__).parent.absolute()


def main():
    img = Image.open(str(PARENT.parent / "test/sample/dog.jpeg"), "r")
    img.thumbnail((512, 512))
    arr = np.array(img)

    joints = json.loads(sys.stdin.read())

    w, h = arr.shape[:2][::-1]

    # RGB to BGR
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Draw all the joints
    for joint in joints["joints"]:
        if joint["confidence"] < 0.5:
            continue

        x = int(joint["x"] * w)
        y = int(joint["y"] * h)
        cv2.circle(arr, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            arr, joint["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

    # Open a window with the plotted image
    # cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.imshow("image", arr)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
