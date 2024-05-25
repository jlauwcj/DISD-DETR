import os
import random
import collections.abc as abc

import numpy as np
import torchvision.transforms as transforms

from e2edet.utils.distributed import is_master
from e2edet.dataset.processor import functional as F


PROCESSOR_REGISTRY = {}


def register_processor(name):
    def register_processor_cls(cls):
        if name in PROCESSOR_REGISTRY:
            raise ValueError("Cannot register duplicate process ({})".format(name))

        PROCESSOR_REGISTRY[name] = cls
        return cls

    return register_processor_cls


def build_processor(config):
    if not hasattr(config, "type"):
        raise AttributeError(
            "Config must have 'type' attribute to specify type of processor"
        )

    if config["type"] in PROCESSOR_REGISTRY:
        processor_class = PROCESSOR_REGISTRY[config["type"]]
    else:
        raise ValueError("Unknown processor type {}".format(config["type"]))

    params = {}
    if not hasattr(config, "params") and is_master():
        print(
            "Config doesn't have 'params' attribute to "
            "specify parameters of the processor "
            "of type {}. Setting to default \{\}".format(config["type"])
        )
    else:
        params = config["params"]

    try:
        processor_instance = processor_class(**params)
    except Exception as e:
        print("Error in", processor_class.__name__)
        raise e

    return processor_instance


class BaseProcessor:
    def __init__(self, params={}):
        for kk, vv in params.items():
            setattr(self, kk, vv)

    def __call__(self, item, *args, **kwargs):
        return item


@register_processor("answer")
class AnswerProcessor(BaseProcessor):
    NO_OBJECT = "<nobj>"

    def __init__(self, class_file, data_root_dir=None):
        defaults = dict(class_file=class_file, data_root_dir=data_root_dir)
        super().__init__(defaults)
        if not os.path.isabs(class_file) and data_root_dir is not None:
            class_file = os.path.join(data_root_dir, class_file)

        if not os.path.exists(class_file):
            raise RuntimeError(
                "Vocab file {} for vocab dict doesn't exist!".format(class_file)
            )

        self.word_list = self._load_str_list(class_file)

    def _load_str_list(self, class_file):
        with open(class_file) as f:
            lines = f.readlines()
        lines = [self._process_answer(l) for l in lines]

        return lines

    def _process_answer(self, answer):
        remove = [",", "?"]
        answer = answer.lower()

        for item in remove:
            answer = answer.replace(item, "")
        answer = answer.replace("'s", " 's")

        return answer.strip()

    def get_size(self):
        return len(self.word_list)

    def idx2cls(self, n_w):
        return self.word_list[n_w]

    def cls2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        else:
            raise ValueError("class %s not in dictionary" % w)

    def __len__(self):
        return len(self.word_list)


# =========================== #
# --------- 2d ops ---------- #
# =========================== #


@register_processor("to_tensor")
class ToTensor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, sample, target=None):
        sample, target = F.to_tensor(sample, target)

        return sample, target


@register_processor("normalize")
class Normalize(BaseProcessor):
    def __init__(self, mean, std, depth_mean=None, depth_std=None):
        defaults = dict(
            mean=mean,
            std=std,
            depth_mean=depth_mean,
            depth_std=depth_std,
        )
        super().__init__(defaults)

    def __call__(self, sample, target=None):
        sample, target = F.normalize(sample, target, mean=self.mean, std=self.std)

        return sample, target


@register_processor("random_size_crop")
class RandomSizeCrop(BaseProcessor):
    def __init__(self, min_size, max_size):
        defaults = dict(min_size=min_size, max_size=max_size)
        super().__init__(defaults)

    def __call__(self, sample, target=None):
        img = sample["image"]
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = transforms.RandomCrop.get_params(img, [h, w])

        return F.crop(sample, target, region)


@register_processor("random_resize")
class RandomResize(BaseProcessor):
    def __init__(self, min_size, max_size=None):
        if isinstance(min_size, int):
            min_size = (min_size,)
        else:
            min_size = list(range(*min_size))
        defaults = dict(min_size=min_size, max_size=max_size)
        super().__init__(defaults)

    def __call__(self, sample, target=None):
        size = random.choice(self.min_size)
        sample, target = F.resize(sample, target, size, self.max_size)

        return sample, target


@register_processor("random_horizontal_flip")
class RandomHorizontalFlip(BaseProcessor):
    def __init__(self, prob=0.5):
        super().__init__(dict(p=prob))

    def __call__(self, sample, target=None):
        if random.random() < self.p:
            sample, target = F.hflip(sample, target)

        return sample, target


@register_processor("random_select")
class RandonSelect(BaseProcessor):
    def __init__(self, preprocessors, probs):
        super().__init__(dict(preprocessors=preprocessors, p=probs))
        self.preprocessors = []
        for preprocessor in preprocessors:
            self.preprocessors.append(build_processor(preprocessor))
        assert len(self.preprocessors) == len(self.p)

    def __call__(self, sample, target=None):
        idx = random.choices(list(range(len(self.preprocessors))), weights=self.p)[0]
        sample, target = self.preprocessors[idx](sample, target)

        return sample, target


@register_processor("resize_scale")
class ResizeScale(BaseProcessor):
    def __init__(self, min_scale, max_scale, image_size):
        super().__init__(
            dict(min_scale=min_scale, max_scale=max_scale, image_size=image_size)
        )

    def __call__(self, sample, target=None):
        scale = random.uniform(self.min_scale, self.max_scale)

        return F.resize_scale(sample, target, scale, self.image_size, self.image_size)


@register_processor("fixed_size_crop")
class FixedSizeCrop(BaseProcessor):
    def __init__(self, image_size, pad_value=0):
        crop_size = (image_size, image_size)
        super().__init__(
            dict(image_size=image_size, pad_value=pad_value, crop_size=crop_size)
        )

    def __call__(self, sample, target=None):
        return F.random_crop(
            sample, target, self.crop_size, is_fixed=True, pad_value=self.pad_value
        )


@register_processor("random_size_crop_v2")
class RandomSizeCropv2(BaseProcessor):
    def __init__(self, image_size):
        crop_size = (image_size, image_size)
        super().__init__(dict(image_size=image_size, crop_size=crop_size))

    def __call__(self, sample, target=None):
        return F.random_crop(sample, target, self.crop_size, is_fixed=False)


