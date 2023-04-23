import config
import torch
import numpy as np
import os
from os.path import join as pjoin
from tools import may_print as mprint
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import functional as TFF
import random
from random import normalvariate as normal
from PIL import Image
from numba import jit
import inspect

nrw_codes = ['aachen', 'bochum', 'dortmund', 'heinsberg', 'muenster', ]


# ========================================================================================== High-level dataset loader

def prepare_training_dataset(cf: config.Config, dom: str, subset: str) -> Dataset:
    """High level function to load a dataset.

    :param cf: Configuration object.
    :param dom: The city to preload.
    :param subset: Code for a subset in ['val' ,'test', 'train', 'all'].
    :return: Dataset in pytorch convention.
    """
    if dom in nrw_codes:
        return DatasetNRWPreloaded(cf, dom, subset)
    else:
        raise NotImplementedError(f"Dataset {dom} not implemented in function '{inspect.stack()[0][3]}'")


# ================================================================================================== DATA AUGMENTATION


class Augmentation:
    def __init__(self, cf: config.Config):
        """ This class provides generic functions for augmentation of images and label maps.

        :param cf: Configuration object.
        """
        self.cf = cf
        self.intpl_mode = TFF.InterpolationMode.BILINEAR if cf.AUG.INTERPOLATE else TFF.InterpolationMode.NEAREST
        self.crop_size = self.cf.TRAIN.IN_SIZE

    def __augmentation_radiometric__(self, sample: dict) -> dict:
        """Apply radiometric augmentation of a data sample.

        :param sample: The sample to augment.
        :return: Augmented sample.
        """
        if self.cf.AUG.RADIO_SCALE == 0:
            return sample

        image, idmap = sample['image'], sample['idmap']

        image = TFF.adjust_brightness(image, brightness_factor=float(np.clip(normal(1, self.cf.AUG.RADIO_SCALE), 0, 2)))
        image = TFF.adjust_contrast(image, contrast_factor=float(np.clip(normal(1, self.cf.AUG.RADIO_SCALE), 0, 2)))

        return {'image': image, 'idmap': idmap}

    def __augmentation_geometric__(self, sample: dict) -> dict:
        """Apply geometric augmentation of a data sample.

        :param sample: The sample to augment.
        :return: Augmented sample.
        """
        image, idmap = sample['image'], sample['idmap']

        if self.cf.AUG.PRECROP:
            pre_crop_size = int(1.5 * self.cf.TRAIN.IN_SIZE)
            h, w = image.size()[1:]
            off_x = random.randint(0, h - pre_crop_size)
            off_y = random.randint(0, w - pre_crop_size)
            image = TFF.crop(image, off_x, off_y, pre_crop_size, pre_crop_size)
            idmap = TFF.crop(idmap, off_x, off_y, pre_crop_size, pre_crop_size)

        if self.cf.AUG.ROTATE:
            angle = random.randint(-180, 180)
            image = TFF.rotate(image, angle, interpolation=self.intpl_mode)
            idmap = TFF.rotate(idmap[None, :, :], angle, fill=self.cf.DATA.IGNORE_INDEX)[0]

        if self.cf.AUG.FLIP:
            if random.random() > 0.5:
                image = TFF.hflip(image)
                idmap = TFF.hflip(idmap)
            if random.random() > 0.5:
                image = TFF.vflip(image)
                idmap = TFF.vflip(idmap)

        if self.cf.AUG.RAND_RESCALE:
            sym = self.cf.AUG.RESCALE_SYM
            h, w = image.size()[1:]
            crop_h = int(random.randint(*self.cf.AUG.RAND_RESCALE) / 100.0 * self.crop_size)
            crop_w = crop_h if sym else int(random.randint(*self.cf.AUG.RAND_RESCALE) / 100.0 * self.crop_size)
            off_x = random.randint(0, h - crop_h)
            off_y = random.randint(0, w - crop_w)
            lcs = [self.crop_size, self.crop_size]
            image = TFF.resized_crop(image, off_x, off_y, crop_h, crop_w, lcs, self.intpl_mode)
            idmap = TFF.resized_crop(idmap, off_x, off_y, crop_h, crop_w, lcs, TFF.InterpolationMode.NEAREST)
        else:
            image = TFF.center_crop(image, self.crop_size)
            idmap = TFF.center_crop(idmap, self.crop_size)

        return {'image': image, 'idmap': idmap}

    def __ul_crop__(self, sample: dict) -> dict:
        """Crop the largest possible up-left area of an image such that after cropping h%32 = w%32 = 0.

        :param sample: The sample to crop from.
        :return: Augmented sample.
        """
        image, idmap = sample['image'], sample['idmap']

        h, w = image.size()[1:]
        ph = h - h % 32
        pw = w - w % 32

        image = TFF.crop(image, 0, 0, ph, pw)
        idmap = TFF.crop(idmap, 0, 0, ph, pw)

        return {'image': image, 'idmap': idmap}

    def __call__(self, sample: dict) -> dict:
        """ Performs augmentation of a sample.

        Note that in this framework, the radiometric augmentation is performed at a later point, because the
        non-augmented samples serve as input for the appearance adaptation network.
        :param sample: The sample to augment.
        :return: Augmented sample.
        """
        sample_aug = self.__augmentation_geometric__(sample)
        sample['image'] = sample_aug['image']
        sample['idmap'] = sample_aug['idmap']
        return sample


# ============================================================================================ GeoNRW Training Dataset

@jit(nopython=True)
def label_mapping_nrw(id_map: np.ndarray) -> np.ndarray:
    """Remaps a 2D label map for the GeoNRW datasets.

    The classes Water, Railway, Highway and Airports_shipyards are mapped to Unknown because they are not represented in some
    cities. Grassland is mapped to Unknown too, because it often shows forest.
    After the mapping, the ids are 0-5, where 5 corresponds to Unknown.
    :param id_map: The 2D map with label ids to process.
    :return: Mapped id map
    """
    h, w = id_map.shape
    for x in range(h):
        for y in range(w):
            v = id_map[x, y]
            if v in [0, 2, 5, 6, 7, 8]:
                o = 5
            elif v == 10:
                o = 0
            elif v == 9:
                o = 2
            else:
                continue
            id_map[x, y] = o
    return id_map


def file_table_nrw(root: str, subset: str) -> pd.DataFrame:
    """Load paths to images and references for a specific city in the GeoNRW dataset.

    :param root: Root folder containing the data for a city.
    :param subset: Code for a subset in ['val' ,'test', 'train', 'all'].
    :return: A pandas Dataframe with columns ['City', 'ImgName', 'Image', 'Ref'].
    """
    data = []
    for img_name in os.listdir(root):
        if not img_name.endswith('rgb.jp2'):
            continue
        last_digit = int(img_name.split('_')[-2]) % 10
        if subset == 'val' and not (last_digit in [0, 1]): continue
        if subset == 'test' and not (last_digit in [2, 3]): continue
        if subset == 'train' and (last_digit in [0, 1, 2, 3]): continue

        p_image = pjoin(root, img_name)
        p_ref = pjoin(root, img_name.replace('rgb.jp2', 'seg.tif'))
        data.append([os.path.split(root)[1], img_name, p_image, p_ref])

    return pd.DataFrame(data, columns=['City', 'ImgName', 'Image', 'Ref'])


def preload_nrw(root: str, dom: str, subset: str) -> tuple:
    """ Preload the data for a city in the GeoNRW dataset.

    :param root: Root to the GeoNRW cities.
    :param dom: The city to preload.
    :param subset: Code for a subset in ['val' ,'test', 'train', 'all'].
    :return: tuple of images, labels and names, each being a list.
    """
    print(f'Pre-loading images for {dom} - {subset}')
    table = file_table_nrw(pjoin(root, dom), subset)
    images, labels, names = [], [], []
    for _, row in table.iterrows():
        city, image_name, img_path, ref_path = row
        image = np.array(Image.open(img_path).convert('RGB'))
        images.append(((image / 127.5) - 1.0).astype(np.float32))
        labelmap = label_mapping_nrw(np.array(Image.open(ref_path)))
        labels.append(labelmap.astype(np.ubyte))
        names.append(f'{city}-{image_name}')

    return images, labels, names


class DatasetNRW(Dataset):
    def __init__(self, cf: config.Config, dom: str, subset: str):
        """ Implementation of pytorch Dataset for a city of the GeoNRW dataset.

        :param cf: Configuration object.
        :param dom: The city to preload.
        :param subset: Code for a subset in ['val' ,'test', 'train', 'all'].
        """
        self.df = file_table_nrw(pjoin(cf.PATHS.GeoNRW, dom), subset)
        self.augmentation = Augmentation(cf)
        self.num_imgs = len(self.df)
        self.ds_length = cf.TRAIN.IT_P_EP * cf.TRAIN.BTSZ

    def __len__(self):
        return self.ds_length

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx % self.num_imgs, 2]
        ref_path = self.df.iloc[idx % self.num_imgs, 3]

        image = np.array(Image.open(img_path).convert('RGB'))
        image = (torch.from_numpy(((image / 127.5) - 1.0).transpose(2, 0, 1))).float()

        label_map = label_mapping_nrw(np.array(Image.open(ref_path)))
        idmap = torch.from_numpy(label_map).long()
        sample = {'image': image, 'idmap': idmap}
        return self.augmentation(sample)


class DatasetNRWPreloaded(Dataset):
    def __init__(self, cf: config.Config, dom, subset):
        """ Implementation of pytorch Dataset for a city of the GeoNRW dataset.

        Equivalent to DatasetNRW, but images and labels are preloaded to memory.
        :param cf: Configuration object.
        :param dom: The city to preload (see dom_codes).
        :param subset: Code for a subset in ['val' ,'test', 'train', 'all'].
        """
        images, labels, _ = preload_nrw(cf.PATHS.GeoNRW, dom, subset)
        self.augment = Augmentation(cf)
        self.num_img = len(images)
        self.ds_length = cf.TRAIN.IT_P_EP * cf.TRAIN.BTSZ

        self.images = [torch.from_numpy(I.transpose(2, 0, 1)).float() for I in images]
        self.refs = [torch.from_numpy(L).long() for L in labels]

    def __len__(self):
        return self.ds_length

    def __getitem__(self, idx):
        sample = {'image': self.images[idx % self.num_img], 'idmap': self.refs[idx % self.num_img]}
        return self.augment(sample)


# ========================================================================================== Generic Evaluation Dataset

class EvalDataset(Dataset):
    def __init__(self, cf: config.Config, dom: str, subset: str):
        """This dataset returns preloaded images in a sliding window based approach.

        :param cf: Configuration object.
        :param dom: Dataset name (see dom_codes).
        :param subset: Subset to use in ['train', 'test', 'val', 'all'].

        __getitem__ will return image batches [n x (h,w,c)] and their coordinates.
        For each patch (image crop / label map) the image id and the coordinates of the upper left corner are stored.
        """

        mprint(f'Creating evaluation dataset for {dom} - {subset}', 1, cf.VRBS)

        if dom in nrw_codes:
            images, labels, names = preload_nrw(cf.PATHS.GeoNRW, dom, subset)
        else:
            raise NotImplementedError(f"Dataset {dom} not implemented in 'EvalDataset'")

        self.images = images
        self.patch_list = []
        self.labels = labels
        self.names = names
        self.size = cf.EVALUATION.IN_SIZE

        # ------------------------------------------------------------------------------ compute list of all sub-patches

        size = cf.EVALUATION.IN_SIZE
        swes = cf.EVALUATION.SW_SHIFT
        for i, image in enumerate(images):
            x = 0
            h, w = image.shape[:2]
            while True:
                y = 0
                xoob = x + size >= h
                while True:
                    yoob = y + size >= w
                    self.patch_list.append((i, x if not xoob else h - size, y if not yoob else w - size))
                    if yoob: break
                    y += swes
                if xoob: break
                x += swes

        self.patch_list = np.array(self.patch_list, dtype=np.int)

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        patch = self.patch_list[idx]
        id = patch[0]
        sample = {'image': self.images[id], 'labels': self.labels[id], 'patch': patch, 'name': self.names[id]}
        return self.transform(sample)

    def transform(self, sample):
        patch = sample['patch']
        name = sample['name']
        _, x, y = patch
        image = sample['image'][x:x + self.size, y:y + self.size, :].transpose((2, 0, 1))
        labels = sample['labels'][x:x + self.size, y:y + self.size]

        assert image.shape[1] == self.size and image.shape[2] == self.size, f"Invalid shape {image.shape} at {patch}"

        return {'image': torch.from_numpy(image).float(),
                'labels': torch.from_numpy(labels).long(),
                'patch': patch,
                'name': name}
