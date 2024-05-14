import json
from pathlib import Path

import numpy as np
from attrs import asdict
from PIL import Image

from human_estimation import Validator

PARENT = Path(__file__).parent.absolute()


def main():
    img = Image.open(str(PARENT.parent / "test/sample/dog.jpeg"), "r")
    arr = np.array(img)

    val = Validator.from_movenet_thunder(min_conf_thres=0.5)
    joints, is_human = val.validate(arr, min_joints=12)

    joint_dict = dict(joints=[asdict(joint) for joint in joints])

    print(json.dumps(joint_dict, indent=4))


if __name__ == "__main__":
    main()
