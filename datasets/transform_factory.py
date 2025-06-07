from typing import Callable, Dict
import torch
from torchvision.transforms import Compose, transforms
from torchvision.transforms.functional import crop, hflip, vflip


class ApplyTransformToKey:
    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


# --------------------- MONUSEG TRANSFORMS ---------------------

def train_monuseg_transform(mean, std):
    def configured_transform(transform_config):
        crop_ = transform_config['crop_']
        top = transform_config['top']
        left = transform_config['left']
        p_hflip = transform_config['hflip']
        p_vflip = transform_config['vflip']
        w_crop = transform_config['w_crop']
        h_crop = transform_config['h_crop']
        corr_type = transform_config['corr_type']
        img_cond = transform_config['img_cond']

        def normalization(img):
            return img

        def normalization_mask(mask):
            return (mask - 0.5) * 2

        def hflip_closure(img):
            return hflip(img) if p_hflip > 0.5 else img

        def vflip_closure(img):
            return vflip(img) if p_vflip > 0.5 else img

        def crop_closure(img):
            return crop(img, top, left, w_crop, h_crop) if crop_ == 1 else img

        return Compose([
            ApplyTransformToKey(
                key="mask",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization,
                    crop_closure,
                    hflip_closure,
                    vflip_closure,
                ]),
            ),
            ApplyTransformToKey(
                key="image",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization_mask,
                    crop_closure,
                    hflip_closure,
                    vflip_closure,
                ]),
            ),
        ])
    return configured_transform


def test_monuseg_transform(mean, std):
    def configured_transform(transform_config):
        crop_ = transform_config['crop_']
        top = transform_config['top']
        left = transform_config['left']
        w_crop = transform_config['w_crop']
        h_crop = transform_config['h_crop']
        corr_type = transform_config['corr_type']
        img_cond = transform_config['img_cond']

        def normalization(img):
            return img

        def normalization_mask(mask):
            return (mask - 0.5) * 2

        def crop_closure(img):
            return crop(img, top, left, w_crop, h_crop) if crop_ == 1 else img

        return Compose([
            ApplyTransformToKey(
                key="mask",
                transform=Compose([
                    transforms.ToTensor(),
                    crop_closure,
                ]),
            ),
            ApplyTransformToKey(
                key="image",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization_mask,
                    crop_closure,
                ]),
            ),
        ])
    return configured_transform


# --------------------- GLAS TRANSFORMS ---------------------

def train_glas_transform(mean, std):
    def configured_transform(transform_config):
        p_hflip = transform_config['hflip']
        p_vflip = transform_config['vflip']
        corr_type = transform_config['corr_type']
        img_cond = transform_config['img_cond']

        def normalization(img):
            return (img - 0.5) * 2 if corr_type == 0 else img

        def normalization_mask(mask):
            return (mask - 0.5) * 2

        def hflip_closure(img):
            return hflip(img) if p_hflip > 0.5 else img

        def vflip_closure(img):
            return vflip(img) if p_vflip > 0.5 else img

        return Compose([
            ApplyTransformToKey(
                key="mask",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization,
                    hflip_closure,
                    vflip_closure,
                ]),
            ),
            ApplyTransformToKey(
                key="image",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization_mask,
                    hflip_closure,
                    vflip_closure,
                ]),
            ),
        ])
    return configured_transform


def test_glas_transform(mean, std):
    def configured_transform(transform_config):
        corr_type = transform_config['corr_type']
        img_cond = transform_config['img_cond']

        def normalization(img):
            return (img - 0.5) * 2 if corr_type == 0 else img

        def normalization_mask(mask):
            return (mask - 0.5) * 2

        return Compose([
            ApplyTransformToKey(
                key="mask",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization,
                ]),
            ),
            ApplyTransformToKey(
                key="image",
                transform=Compose([
                    transforms.ToTensor(),
                    normalization_mask,
                ]),
            ),
        ])
    return configured_transform


# --------------------- FACTORY + UTILS ---------------------

def inv_normalize(mean, std):
    return transforms.Normalize(mean=-mean / std, std=1 / std)


def transform_factory(cfg):
    if cfg.modality == 'monuseg':
        return {
            'train': train_monuseg_transform,
            'test': test_monuseg_transform
        }
    elif cfg.modality == 'glas':
        return {
            'train': train_glas_transform,
            'test': test_glas_transform
        }
    else:
        raise ValueError(f"Unknown modality '{cfg.modality}' specified!")
