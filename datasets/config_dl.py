from types import SimpleNamespace
from torch.utils.data.dataloader import DataLoader
import yaml
import json 
from os.path import join, isfile

from datasets.transform_factory import inv_normalize, transform_factory
from datasets.monuseg_dataset import MoNuSegDataset
from datasets.glas_dataset import GlaSDataset

def config_dl(cfg):
    if cfg.general.modality == 'monuseg':
        DatasetType = MoNuSegDataset
        stats = {'mean': 0., 'std': 1.}

    elif cfg.general.modality == 'glas':
        DatasetType = GlaSDataset
        stats = {'mean': 0., 'std': 1.}
        
    else:
        raise ValueError('Unknown modality %s specified!' %cfg.modality)

    train_dataset   = DatasetType(cfg.general.data_path, f'{cfg.general.data_path}/{cfg.general.csv_train}', cfg=cfg.general, batch_size=cfg.general.batch_size)
    test_dataset    = DatasetType(cfg.general.data_path, f'{cfg.general.data_path}/{cfg.general.csv_test}', cfg=cfg.general, batch_size=cfg.inference.n_samples, test_mode=True)    

    train_dataloader     = DataLoader(train_dataset, batch_size=cfg.general.batch_size, shuffle=True, drop_last=True)
    test_dataloader      = DataLoader(test_dataset, batch_size=cfg.inference.n_samples, shuffle=True, drop_last=False) # TODO: True/False

    dbdict = {"train_dl": train_dataloader, "test_dl": test_dataloader}
        
    train_dl, test_dl = dbdict["train_dl"], dbdict["test_dl"]
    tfdict = transform_factory(cfg.general)
    T_train, T_test = tfdict["train"](stats["mean"], stats["std"]), tfdict["test"](
        stats["mean"], stats["std"]
    )
    train_dl.dataset.transform, test_dl.dataset.transform = T_train, T_test
    train_dl.inv_normalize, test_dl.inv_normalize = inv_normalize(
        stats["mean"], stats["std"]
    ), inv_normalize(stats["mean"], stats["std"])

    return train_dl, test_dl
