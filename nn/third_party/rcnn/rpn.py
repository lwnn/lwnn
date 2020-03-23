# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>

import math
import numpy as np

__all__ = ['generate_pyramid_anchors', 'proposal_forward']

# code copied from https://github.com/matterport/Mask_RCNN 
def compute_backbone_shapes(config, image_shape):
        return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config['BACKBONE_STRIDES']])

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes

def _generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)

def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)

def generate_pyramid_anchors(config):
    backbone_shapes = compute_backbone_shapes(config, config['IMAGE_SHAPE'])
    a = _generate_pyramid_anchors(config['RPN_ANCHOR_SCALES'],
                                             config['RPN_ANCHOR_RATIOS'],
                                             backbone_shapes,
                                             config['BACKBONE_STRIDES'],
                                             config['RPN_ANCHOR_STRIDE'])
    anchors = norm_boxes(a, config['IMAGE_SHAPE'][:2])
    return anchors

def TopK(X, k, axis=-1, largest=True):
    # ref https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/node/topk.py
    sorted_indices = np.argsort(X, axis=axis) # order from small to big
    sorted_values = np.sort(X, axis=axis)
    if(largest==True):
        sorted_indices = np.flip(sorted_indices, axis=axis) # order frim big to small
        sorted_values = np.flip(sorted_values, axis=axis)
    topk_sorted_indices = np.take(sorted_indices, np.arange(k), axis=axis)
    topk_sorted_values = np.take(sorted_values, np.arange(k), axis=axis)
    return topk_sorted_values, topk_sorted_indices

# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    batch_size = inputs[0].shape[0]
    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    result = [np.stack(o, axis=0) for o in outputs]
    if len(result) == 1:
        result = result[0]

    return result

def Gather(data, indices, axis=0):
    return np.take(data, indices, axis=axis)

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = np.stack([y1, x1, y2, x2], axis=1)
    return result

def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = np.split(window, 4)
    y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
    # Clip
    y1 = np.maximum(np.minimum(y1, wy2), wy1)
    x1 = np.maximum(np.minimum(x1, wx2), wx1)
    y2 = np.maximum(np.minimum(y2, wy2), wy1)
    x2 = np.maximum(np.minimum(x2, wx2), wx1)
    clipped = np.concatenate([y1, x1, y2, x2], axis=1)
    clipped.reshape((clipped.shape[0], 4))
    return clipped

def non_max_suppression_fast(boxes, scores, keep, nms_threshold):
    # ref https://github.com/bruceyang2012/nms_python/blob/master/nms.py
    # initialize the list of picked indexes
    indices = []

    # grab the coordinates of the bounding boxes
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1) * (y2 - y1)

    # sort the indexes
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        indices.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > nms_threshold)[0])))

    return indices[:keep]

def proposal_forward(RPN_BBOX_STD_DEV, scores, deltas, anchors, proposal_count, nms_threshold):
    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = scores[:, :, 1]
    deltas = deltas * np.reshape(RPN_BBOX_STD_DEV, [1, 1, 4])
    pre_nms_limit = min(6000, anchors.shape[1])
    _,ix = TopK(scores, pre_nms_limit, largest=True)
    scores = batch_slice([scores, ix], lambda x, y: Gather(x, y))
    deltas = batch_slice([deltas, ix], lambda x, y: Gather(x, y))
    pre_nms_anchors = batch_slice([anchors, ix], lambda a, x: Gather(a, x))

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = batch_slice([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y))

    # Clip to image boundaries. Since we're in normalized coordinates,
    # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
    window = np.array([0, 0, 1, 1], dtype=np.float32)
    boxes = batch_slice(boxes, lambda x: clip_boxes_graph(x, window))
    def nms(boxes, scores):
        indices = non_max_suppression_fast(boxes, scores, proposal_count, nms_threshold)
        proposals = Gather(boxes, indices)
        return proposals
    proposals = batch_slice([boxes, scores], nms)
    return proposals