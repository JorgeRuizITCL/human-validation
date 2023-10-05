from typing import Tuple

import numpy as np
import pytest

from ..img import resize_pad_crop


def create_img(res: Tuple[int, int]):
    # res = w, h
    return np.random.randint(low=0, high=255, size=(*res, 3)).astype(np.uint8)



def test_crop_pad_no_change():
    rgb_image = create_img((10, 10))
    
    res = resize_pad_crop(rgb_image, resolution=(10, 10))
    
    np.testing.assert_equal(res, rgb_image)


def test_crop_pad_downsize():
    
    rgb_img = create_img((10, 10))
    res = resize_pad_crop(rgb_img, resolution=(5, 5))
    
    assert res.shape == (5,5, 3)
    mean = np.mean(res)
    assert  mean > 0
    assert mean <= 255
    
def test_v_pad():
    rgb_img = create_img((20, 10))
    res = resize_pad_crop(rgb_img, resolution=(20, 20))
    assert res.shape == (20, 20, 3)
    assert np.sum(res[:, :5]) == 0 # filled with black
    assert np.sum(res[:,-5:]) == 0
    
    assert np.sum(res[:,5:15]) > 0

def test_h_pad():
    
    rgb_img = create_img((10, 20))
    res = resize_pad_crop(rgb_img, resolution=(20, 20))
    assert res.shape == (20,20, 3)
    assert np.sum(res[:5,]) == 0 # filled with black
    assert np.sum(res[-5:]) == 0
    
    assert np.sum(res[5:15]) > 0
    
def test_crop_pad():
    rgb_img = create_img((10, 20))
    res = resize_pad_crop(rgb_img, resolution=(5, 5))
    assert res.shape == (5,5,3)
    print(res)
    assert np.sum(res[:1,]) == 0 # filled with black
    assert np.sum(res[-1:]) == 0
    

    