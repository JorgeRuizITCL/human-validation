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

DEFAULT_PROVIDERS =  [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'}), ('CUDAExecutionProvider',)]

class Validator:
    
    def __init__(self, model_path: str, onnx_providers: Any = None, min_conf_thres: float = 0.66) -> None:
        onnx_providers = DEFAULT_PROVIDERS if onnx_providers is None else onnx_providers
        self._conf_thres = min_conf_thres
        self._sess = ort.InferenceSession(model_path, providers=onnx_providers)
        input_node = self._sess.get_inputs()[0]
        self._input_name, self._input_shape = input_node.name, input_node.shape
    
    def from_movenet_lightning():
        path = _LIB_PTH / "models/onnx/lite-model_movenet_singlepose_lightning_3_op15.onnx"
        return Validator(str(path))
    
    def from_movenet_thunder():
        path = _LIB_PTH / "models/onnx/lite-model_movenet_singlepose_thunder_3_op15.onnx"
        return Validator(str(path))
        
    def validate(self, img_rgb_u8: np.ndarray, min_joints: int) -> Tuple[List[Joint], bool]:
        reshaped_img = resize_pad_crop(img_rgb_u8, self._input_shape[-3:-1])
        
        reshaped_img = np.expand_dims(reshaped_img, 0) # add batch dim
        
        
        norm_img = (reshaped_img).astype(np.float32)
        
        res = self._sess.run(None, {self._input_name: norm_img})[0]
        
        joints = decode_movenet(res, self._conf_thres)
        
        n_vis_joints = sum(int(j.is_visible) for j in joints)
        print(n_vis_joints)
        return joints, n_vis_joints >= min_joints