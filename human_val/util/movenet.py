from typing import List, cast

import numpy as np

from .joint import Joint

NOSE = "nose"
LEFT_EYE = "left_eye"
RIGHT_EYE = "right_eye"
LEFT_EAR = "left_ear"
RIGHT_EAR = "right_ear"
LEFT_SHOULDER = "left_shoulder"
RIGHT_SHOULDER = "right_shoulder"
LEFT_ELBOW = "left_elbow"
RIGHT_ELBOW = "right_elbow"
LEFT_WRIST = "left_wrist"
RIGHT_WRIST = "right_wrist"
LEFT_HIP = "left_hip"
RIGHT_HIP = "right_hip"
LEFT_KNEE = "left_knee"
RIGHT_KNEE = "right_knee"
LEFT_ANKLE = "left_ankle"
RIGHT_ANKLE = "right_ankle"

_ids = [
    NOSE,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]

# 06D6A0
GREEN = (160, 214, 6)
# 1B9AAA
BLUE = (170, 154, 27)
# 8DD55A
YELLOW_GREEN = (90, 213, 141)
# 85718d
OLD_LAVENDER = (
    141,
    113,
)
# EF476F
PINK = (111, 71, 239)
# EB7745
ORANGE = (69, 119, 235)
# FBA54A
RAJAH = (74, 165, 251)
# 8AFA5A
GREEN = (90, 250, 138)
# E21515
RED = (21, 21, 226)
# WHITE
WHITE = (255, 255, 255)

MOVENET_CONNECTIONS = [
    (NOSE, LEFT_EYE, GREEN),
    (LEFT_EYE, LEFT_EAR, GREEN),
    (NOSE, RIGHT_EYE, YELLOW_GREEN),
    (RIGHT_EYE, RIGHT_EAR, YELLOW_GREEN),
    (NOSE, LEFT_SHOULDER, ORANGE),
    (LEFT_SHOULDER, LEFT_ELBOW, RAJAH),
    (LEFT_ELBOW, LEFT_WRIST, PINK),
    (LEFT_SHOULDER, LEFT_HIP, OLD_LAVENDER),
    (LEFT_HIP, LEFT_KNEE, BLUE),
    (LEFT_KNEE, LEFT_ANKLE, RED),
    (NOSE, RIGHT_SHOULDER, ORANGE),
    (RIGHT_SHOULDER, RIGHT_ELBOW, RAJAH),
    (RIGHT_ELBOW, RIGHT_WRIST, PINK),
    (RIGHT_SHOULDER, RIGHT_HIP, OLD_LAVENDER),
    (RIGHT_HIP, RIGHT_KNEE, BLUE),
    (RIGHT_KNEE, RIGHT_ANKLE, RED),
    # Joints
    (LEFT_SHOULDER, RIGHT_SHOULDER, WHITE),
    (LEFT_HIP, RIGHT_HIP, WHITE),
]


def decode_movenet(
    output: np.ndarray, threshold: float
) -> List[Joint]:
    """Transforms the output of a 1,1,17,3 movenet output tensor

    Args:
        output (np.ndarray): Network output
        threshold (float): Min threshold to consider a joint visible.

    Returns:
        list[Joint]: List of joints
    """
    assert output.shape == (1, 1, 17, 3)
    joints: List[Joint] = []
    for i in range(output.shape[0]): # For whatever this dimension is
        for j in range(output.shape[2]): #for each joint in img
            confidence = output[i, 0, j, 2]
            joints.append(
                Joint(
                    id=j,
                    label=_ids[j],
                    confidence=confidence,
                    threshold=threshold,
                )
            )

    return joints