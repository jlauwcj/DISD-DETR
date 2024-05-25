import math

import torch
import torchvision.transforms.functional as F
import numpy as np

from e2edet.utils.box_ops import box_xyxy_to_cxcywh
from e2edet.utils.general import interpolate, normalize_period, data_to_tensor


# =========================== #
# --------- 2d ops ---------- #
# =========================== #


def resize_scale(sample, target, scale, target_height, target_width):
    def _get_resize(image_size, scale):
        w, h = image_size
        input_size = torch.tensor([h, w])

        # Compute new target size given a scale
        target_size = torch.tensor([target_height, target_width])
        target_scale_size = target_size * scale

        # Compute actual rescaling applied to input image and output size
        output_scale = torch.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = torch.round(input_size * output_scale).int()
        oh, ow = output_size.tolist()

        return (ow, oh)

    size = _get_resize(sample["image"].size, scale)

    return resize(sample, target, size)


def random_crop(sample, target, crop_size, is_fixed=True, pad_value=0):
    def _get_crop(image_size, crop_size, is_fixed):
        w, h = image_size
        ow, oh = crop_size

        # Add random crop if the image is scaled up
        max_offset = torch.tensor([h, w]) - torch.tensor([oh, ow])
        max_offset = torch.clamp(max_offset, min=0)

        offset = max_offset * np.random.uniform(0.0, 1.0)
        offset = torch.round(offset).int().tolist()

        if is_fixed:
            return (offset[0], offset[1], oh, ow)

        return (offset[0], offset[1], min(oh, h), min(ow, w))

    size = _get_crop(sample["image"].size, crop_size, is_fixed)

    if is_fixed:
        w, h = sample["image"].size
        ow, oh = crop_size

        pad_size = torch.tensor([oh, ow]) - torch.tensor([h, w])
        pad_size = torch.clamp(pad_size, min=0).tolist()
        sample, target = pad(
            sample, target, (pad_size[1], pad_size[0]), pad_value=pad_value
        )

    return crop(sample, target, size)


def crop(sample, target, region):
    """
    Crop region in the image. For 3D annotations, it considers their 2D projection on image.
    """
    # region: [y, x, h, w]
    cropped_image = F.crop(sample["image"], *region)

    i, j, h, w = region

    target = target.copy()
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]  # x1, y1, x2, y2
        max_size = torch.tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)

        area = (cropped_boxes[:, 1] - cropped_boxes[:, 0]).prod(dim=1)
        target["boxes"] = cropped_boxes.view(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].view(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1] > cropped_boxes[:, 0], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    sample["image"] = cropped_image

    return sample, target


def hflip(sample, target):
    flipped_image = F.hflip(sample["image"])

    w, h = sample["image"].size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor(
            [w, 0, w, 0]
        )
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    sample["image"] = flipped_image

    return sample, target


def pad(sample, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(sample["image"], (0, 0, padding[0], padding[1]))

    if target is None:
        sample["image"] = padded_image

        return sample, None

    target = target.copy()
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(
            target["masks"], (0, padding[0], 0, padding[1])
        )

    sample["image"] = padded_image

    return sample, target


def resize(sample, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def _get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_orig_size = float(min((w, h)))
            max_orig_size = float(max((w, h)))
            if max_orig_size / min_orig_size * size > max_size:
                size = int(round(max_size * min_orig_size / max_orig_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def _get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return _get_size_with_aspect_ratio(image_size, size, max_size)

    size = _get_size(sample["image"].size, size, max_size)
    rescaled_image = F.resize(sample["image"], size)

    if target is None:
        sample["image"] = rescaled_image

        return sample, None

    ratios = tuple(
        float(s) / float(s_orig)
        for s, s_orig in zip(rescaled_image.size, sample["image"].size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0]
            > 0.5
        )

    sample["image"] = rescaled_image

    return sample, target


def to_tensor(sample, target):
    sample["image"] = F.to_tensor(sample["image"])

    return sample, target


def normalize(sample, target, mean, std):
    sample["image"] = F.normalize(sample["image"], mean=mean, std=std)
    if target is None:
        return sample, None

    target = target.copy()
    h, w = sample["image"].shape[-2:]
    if "boxes" in target:
        boxes = target["boxes"]
        target["orig_boxes"] = boxes
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        target["boxes"] = boxes

    return sample, target
