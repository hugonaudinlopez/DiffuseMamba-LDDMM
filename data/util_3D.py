import os
import torch
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']




def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWD to 1DHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 1, 0)))).float().unsqueeze(0)
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


def augment(img_list, hflip=True, rot=False, split='val'):
    # horizontal flip OR rotate
    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        return img
    return [_augment(img) for img in img_list]


