from pathlib import Path

import numpy as np
from PIL import Image

from human_val import Validator

_FILE_PTH = Path(__file__).parent
SEED = 0
np.random.seed(SEED)


def test_validator():
    noise_img = np.random.randint(low=0, high=255, size=(500, 500, 3)).astype(np.uint8)

    val = Validator.from_movenet_lightning()

    joints, is_human = val.validate(noise_img, 1)
    assert not is_human

    img = Image.open(str(_FILE_PTH / "sample/sample_pic.jpeg"), "r")

    img_arr = np.array(img)

    joints, is_human = val.validate(img_arr, 12)
    assert is_human
