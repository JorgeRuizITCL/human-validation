from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from human_val.util.img import resize_pad_crop
from human_val.util.joint import Joint
from human_val.util.movenet import decode_movenet

_FILE_PTH = Path(__file__).parent.absolute()
_LIB_PTH = _FILE_PTH.parent


DEFAULT_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


class Validator:
    """
    Human in image validator
    """

    def __init__(
        self, model_path: str, onnx_providers: Any = None, min_conf_thres: float = 0.5
    ) -> None:
        """_summary_

        Args:
            model_path (str): ONNX Movenet model path
            onnx_providers (Any, optional): ONNX Provider, format must match
                https://onnxruntime.ai/docs/execution-providers/. Defaults to None.
            min_conf_thres (float, optional): Min score threshold.
                If the joint score is over the threshold the joint is considered visible.
                Defaults to 0.5, Google uses in an example 0.15
        """
        onnx_providers = DEFAULT_PROVIDERS if onnx_providers is None else onnx_providers
        self._conf_thres = min_conf_thres
        self._sess = ort.InferenceSession(model_path, providers=onnx_providers)
        input_node = self._sess.get_inputs()[0]
        self._input_name, self._input_shape = input_node.name, input_node.shape

    def from_movenet_lightning(onnx_providers: Any = None, min_conf_thres: float = 0.5):
        """Creates a validator based on Movenet Lightning.
        This is the fastest lightning
        Args:
            onnx_providers (Any, optional): Same as the __init__. Defaults to None.
            min_conf_thres (float, optional): Same as the __init__. Defaults to 0.5.

        Returns:
            Validator: An instance of the validator
        """
        path = (
            _LIB_PTH / "models/onnx/lite-model_movenet_singlepose_lightning_3_op15.onnx"
        )
        return Validator(
            str(path), onnx_providers=onnx_providers, min_conf_thres=min_conf_thres
        )

    def from_movenet_thunder(onnx_providers: Any = None, min_conf_thres: float = 0.5):
        """Creates a validator based on Movenet Thunder.
        Most accurate model
        Args:
            onnx_providers (Any, optional): Same as the __init__. Defaults to None.
            min_conf_thres (float, optional): Same as the __init__. Defaults to 0.5.

        Returns:
            Validator: An instance of the validator
        """
        path = (
            _LIB_PTH / "models/onnx/lite-model_movenet_singlepose_thunder_3_op15.onnx"
        )
        return Validator(
            str(path), onnx_providers=onnx_providers, min_conf_thres=min_conf_thres
        )

    def validate(
        self, img_rgb_u8: np.ndarray, min_joints: int
    ) -> Tuple[List[Joint], bool]:
        """Validates a RGB numpy tensor

        Args:
            img_rgb_u8 (np.ndarray): Numpy RGB image in uint8 and shape W,H,3
            min_joints (int): Number of min joints that are visible to consider the
                human visible.

        Returns:
            Tuple[List[Joint], bool]: List of joints and if there is a human in the img.
        """
        img_res_wh: Tuple[int, int] = img_rgb_u8.shape[:2][::-1]  # type: ignore

        reshaped_img = resize_pad_crop(img_rgb_u8, self._input_shape[-3:-1])

        reshaped_img = np.expand_dims(reshaped_img, 0)  # add batch dim

        norm_img = (reshaped_img).astype(np.float32)

        res = self._sess.run(None, {self._input_name: norm_img})[0]

        joints = decode_movenet(res, self._conf_thres, img_res_wh)

        n_vis_joints = sum(int(j.is_visible) for j in joints)

        return joints, n_vis_joints >= min_joints
