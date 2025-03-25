import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFAULT_ROOT = "./dataset/oxford-iiit-pet"


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root=DEFAULT_ROOT, mode="train"):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)


        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        
        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)) # d: default        
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))
        
        image = normalize(image)
        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


# Augmented Dataset
class AugDataset(OxfordPetDataset):

    def __init__(self, type="combine"):
        super().__init__(mode="train")
        if type == "combine":
            self.aug_transform = A.Compose([
                A.Affine(
                    translate_percent=(0.03, 0.03), 
                    rotate=(-10, 10),
                    p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5)
            ])
        elif type == "hf":
            self.aug_transform = A.HorizontalFlip(p=1.0)
        elif type == "vf":
            self.aug_transform = A.VerticalFlip(p=1.0)
        else:
            assert False, "unknown type of AugDataset"


    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)) # d: default        
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        augmented  = self.aug_transform(image=image, mask=mask)
        # print(augmented)
        # print(type(augmented))
        image = augmented['image']
        mask = augmented['mask']

        image = normalize(image)
        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)



def load_dataset(data_path=DEFAULT_ROOT, mode='train', n_aug=0):
    dataset = SimpleOxfordPetDataset(data_path, mode)
    if mode == "train" and n_aug > 0:
        for _ in range(n_aug):
            aug_dataset = AugDataset(type="combine")
            dataset += aug_dataset

    # print(len(dataset))

    return dataset

def load_fliped_augmented_train_dataset(data_path=DEFAULT_ROOT, n_aug=1):
    n_aug = 2 if n_aug > 2 else n_aug

    # all flips: hf, vf
    flips=['hf', 'vf']
    dataset = SimpleOxfordPetDataset(data_path, "train")
    for i in range(n_aug):
        aug_dataset = AugDataset(type=flips[i])
        dataset += aug_dataset


    # print(len(dataset))

    return dataset




if __name__ == "__main__":
    
    # use this to download raw data first
    OxfordPetDataset.download('./dataset/oxford-iiit-pet')


    # from utils import new_writer
    # writer = new_writer(type="dataset")

    # # print(len(load_dataset(mode="valid")))
    # # print(len(load_dataset(mode="test")))
    # # print(len(load_dataset(mode='train')))
    # # print(len(load_dataset(mode='train', n_aug=1)))

    # # a_t = load_dataset(mode='train', n_aug=1)
    # # for i in range(len(a_t)):
    # #     print(a_t[i]['image'].shape)
    
    
    # # dataset = SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='test')
    # # dataset = SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='train')
    
    # a_dataset = load_fliped_augmented_train_dataset(n_aug=1)
    # # print(test_dataset[0]['trimap'].tolist())
    # # a = len(a_dataset)

    # for i in range(100):
    #     d = a_dataset[-i]
    #     # print(d['image'].shape)
    #     # print(d['mask'].shape)
    #     writer.add_image('aug_image', d['image'], i)
    #     writer.add_image('aug_mask', d['mask'], i)
    #     # writer.add_image('trimap', d['trimap'], i)

    # n_dataset = load_dataset()
    # # n = len(n_dataset)
    # for i in range(100):
    #     d = n_dataset[-i]
    #     # print(d['image'].shape)
    #     # print(d['mask'].shape)
    #     writer.add_image('orig_img', d['image'], i)
    #     writer.add_image('orig_mask', d['mask'], i)
    #     # writer.add_image('trimap', d['trimap'], i)


    pass

