# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>

import math
import numpy as np

__all__ = ['generate_pyramid_anchors']

try:
    from mrcnn import utils
except ImportError as e:
    raise Exception('please export PYTHONPATH=$PYTHONPATH:/path/to/Mask_RCNN for linux\n'
                    '       set PYTHONPATH=%PYTHONPATH%;/path/to/Mask_RCNN for windows', e)

# code copied from https://github.com/matterport/Mask_RCNN 
def compute_backbone_shapes(config, image_shape):
        return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config['BACKBONE_STRIDES']])

def generate_pyramid_anchors(config):
    backbone_shapes = compute_backbone_shapes(config, config['IMAGE_SHAPE'])
    a = utils.generate_pyramid_anchors(config['RPN_ANCHOR_SCALES'],
                                             config['RPN_ANCHOR_RATIOS'],
                                             backbone_shapes,
                                             config['BACKBONE_STRIDES'],
                                             config['RPN_ANCHOR_STRIDE'])
    anchors = utils.norm_boxes(a, config['IMAGE_SHAPE'][:2])
    return anchors.astype(np.float32)
