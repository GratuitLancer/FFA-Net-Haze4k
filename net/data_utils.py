"""Paired Haze4K/RESIDE datasets and explicitly constructed data loaders."""

import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
HAZE_MEAN = (0.64, 0.6, 0.58)
HAZE_STD = (0.14, 0.15, 0.152)


def normalize_haze(image):
    return TF.normalize(TF.to_tensor(image), HAZE_MEAN, HAZE_STD)


def image_paths(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS)


class _PairedDataset(Dataset):
    def __init__(self, pairs, train, size):
        if not pairs:
            raise ValueError("No paired images found")
        if not isinstance(size, str) and (not isinstance(size, int) or size <= 0):
            raise ValueError("Crop size must be a positive integer or 'whole_img'")
        self.pairs = pairs
        self.train = train
        self.size = size
        self.haze_imgs = [pair[0] for pair in pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        haze_path, clear_path = self.pairs[index]
        with Image.open(haze_path) as source:
            haze = source.convert("RGB")
        with Image.open(clear_path) as source:
            clear = source.convert("RGB")
        # RESIDE clear images can include a border; retain the original alignment.
        clear = TF.center_crop(clear, (haze.height, haze.width))
        if isinstance(self.size, int):
            # Upscale small pairs together instead of repeatedly resampling files.
            # The old resampling loop could hang if every image was too small.
            scale = max(1.0, self.size / haze.width, self.size / haze.height)
            if scale > 1:
                dimensions = (math.ceil(haze.width * scale), math.ceil(haze.height * scale))
                haze = haze.resize(dimensions, Image.Resampling.BILINEAR)
                clear = clear.resize(dimensions, Image.Resampling.BILINEAR)
            top, left, height, width = transforms.RandomCrop.get_params(haze, (self.size, self.size))
            haze = TF.crop(haze, top, left, height, width)
            clear = TF.crop(clear, top, left, height, width)
        return self.augData(haze, clear)

    def augData(self, image, target):
        if self.train:
            if random.randint(0, 1):
                image, target = TF.hflip(image), TF.hflip(target)
            rotation = random.randint(0, 3) * 90
            if rotation:
                image, target = TF.rotate(image, rotation), TF.rotate(target, rotation)
        return normalize_haze(image), TF.to_tensor(target)


class RESIDE_Dataset(_PairedDataset):
    def __init__(self, path, train, size=240, format=".png"):
        path = Path(path)
        self.clear_dir = path / "clear"
        self.format = format
        pairs = []
        for haze in image_paths(path / "hazy"):
            clear = self.clear_dir / (haze.stem.split("_")[0] + format)
            if not clear.is_file():
                raise FileNotFoundError(f"Missing clear image for {haze.name}: {clear}")
            pairs.append((haze, clear))
        super().__init__(pairs, train, size)


class Haze4K_Dataset(_PairedDataset):
    def __init__(self, path, train, size=240):
        self.path = Path(path)
        self.haze_dir, self.gt_dir = self.path / "haze", self.path / "gt"
        self.gt_imgs = {image.stem: image for image in image_paths(self.gt_dir)}
        pairs = []
        for haze in image_paths(self.haze_dir):
            clear_id = haze.stem.split("_")[0]
            if clear_id not in self.gt_imgs:
                raise FileNotFoundError(f"Missing gt image for {haze.name} in {self.gt_dir}")
            pairs.append((haze, self.gt_imgs[clear_id]))
        super().__init__(pairs, train, size)


DATASET_SPECS = {
    "its_train": (RESIDE_Dataset, "RESIDE/ITS", True, ".png"),
    "its_test": (RESIDE_Dataset, "RESIDE/SOTS/indoor", False, ".png"),
    "ots_train": (RESIDE_Dataset, "RESIDE/OTS", True, ".jpg"),
    "ots_test": (RESIDE_Dataset, "RESIDE/SOTS/outdoor", False, ".png"),
    "haze4k_train": (Haze4K_Dataset, "Haze4K/train", True, None),
    "haze4k_test": (Haze4K_Dataset, "Haze4K/test", False, None),
}


def build_loader(dataset_cls, dataset_path, train, batch_size, size, num_workers=0, **kwargs):
    dataset = dataset_cls(dataset_path, train=train, size=size, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


def create_loader(name, data_root, batch_size=16, crop_size=240, num_workers=0):
    try:
        dataset_cls, relative_path, train, image_format = DATASET_SPECS[name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset: {name}") from error
    kwargs = {"format": image_format} if image_format is not None else {}
    return build_loader(
        dataset_cls, Path(data_root) / relative_path, train=train,
        batch_size=batch_size if train else 1,
        size=crop_size if train else "whole_img",
        num_workers=num_workers, **kwargs,
    )


def create_loaders(options):
    crop_size = options.crop_size if options.crop else "whole_img"
    return (
        create_loader(options.trainset, options.data_root, options.bs, crop_size, options.num_workers),
        create_loader(options.testset, options.data_root, num_workers=options.num_workers),
    )


def tensorShow(tensors, titles=None):
    """Retain the original optional image-display helper."""
    from matplotlib import pyplot as plt
    from torchvision.utils import make_grid

    titles = titles or [str(index) for index in range(len(tensors))]
    figure = plt.figure()
    for index, (tensor, title) in enumerate(zip(tensors, titles), 1):
        axis = figure.add_subplot(len(tensors), 1, index)
        axis.imshow(make_grid(tensor.detach().cpu()).permute(1, 2, 0).numpy())
        axis.set_title(title)
    plt.show()
