__all__ = ['get_dataloader']
import os
import numpy as np
import volumentations as V
import cv2
import random
from PIL import Image
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader
from ..process import *
IMG_SIZE = (128, 128)

class BHEDataset(Dataset):
    def __init__(
        self, mode, data_list, data_dir, label_norm, augment=True,
        s1_index_list='all',
        s2_index_list='all', months_list='all'
    ):
        super(BHEDataset, self).__init__()
        self.mode           = mode
        self.augment        = augment
        self.transform      = None
        self.data_list      = data_list
        self.data_dir       = data_dir
        self.label_norm     = label_norm
        if self.augment:
            self.transform = V.Compose([
                V.Flip3d(0, p=0.5),
                V.Flip3d(1, p=0.5),
                V.Flip3d(2, p=0.5),
            ], p=1.0)
        self.months_list = months_list
        self.s1_index_list = s1_index_list
        if s1_index_list == 'all':
            self.s1_index_list = list(range(4))
        
        self.s2_index_list = s2_index_list
        if s2_index_list == 'all':
            self.s2_index_list = list(range(11))
    def __len__(self):
        return len(self.data_list)

    def _load_data(self, subject_path):
        subject = os.path.basename(subject_path)
        # loads label data
        label_path = opj(self.data_dir, 'LABEL_1m', subject_path)
        assert os.path.isfile(label_path), f'label {label_path} is not exist'
        label = read_raster(label_path, True, GT_SHAPE)
        label = imread(label_path)
        label = np.nan_to_num(label)
        label = cv2.resize(label, (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
        label[label<1.0]=0.0
        
        target_seg = label.copy()
        target_seg[target_seg<1.0]=0.0
        target_seg[target_seg>1.0]=1.0
        target_seg = np.expand_dims(target_seg, axis=0)
        target_seg = np.expand_dims(target_seg, axis=-1)

        if self.label_norm:
            label = normalize(label)
        label = np.expand_dims(label, axis=0)
        label = np.expand_dims(label, axis=-1)

        # loads S1 and S2 features
        feature_list, mask = [], []
        for month in self.months_list:
            file_name = '%s_%02d.tif' % (str.split(subject_path, '.')[0], month)
            s1_path = opj(self.data_dir, 'S1', file_name)
            s2_path = opj(self.data_dir, 'S2', file_name)
            
            img_s1 = GRD_toRGB_S1(s1_path)
            s1_list = img_s1.astype("float32")
            img_s2 = GRD_toRGB_S2(s2_path)
            s2_list = img_s2.astype("float32")

            feature = np.concatenate([s1_list , s2_list], axis=-1)
            feature = np.expand_dims(feature, axis=0)
            feature_list.append(feature)
            mask.append(False)
        feature = np.concatenate(feature_list, axis=0)
        mask = np.array(mask)
        return label, target_seg, feature, mask

    def __getitem__(self, index):
        subject_path = self.data_list[index]
        label, target_seg, feature, mask = self._load_data(subject_path)
        
        # NO FLIP only channel drop
        if self.augment:
            data = {'image': feature, 'mask1': label, 'mask2': target_seg}
            aug_data = self.transform(**data)
            feature, label, target_seg = aug_data['image'], aug_data['mask1'], aug_data['mask2']
            if label.shape[0] > 1:
                label = label[:1]
            if random.random() > 0.5: 
                while True:
                    mask2 = np.random.rand(*mask.shape) < 0.3
                    mask3 = np.logical_or(mask, mask2)
                    if not mask3.all():
                        break
                mask = mask3
                feature[mask2] = 0

        feature = feature.transpose(0, 3, 1, 2).astype(np.float32)
        label = label[0].transpose(2, 0, 1).astype(np.float32)
        target_seg = target_seg[0].transpose(2, 0, 1).astype(np.float32)        
        return feature, mask, label, target_seg

def get_dataloader(
    mode, data_list, data_dir, label_norm, configs
):
    assert mode in ['train', 'val']
    if mode == 'train':
        batch_size = configs.train_batch
        drop_last  = True
        shuffle    = True
        augment    = configs.apply_augment
    else:  # mode == 'val'
        batch_size = configs.val_batch
        drop_last  = False
        shuffle    = False
        augment    = False
    dataset = BHEDataset(
        mode           = mode,
        data_list      = data_list,
        data_dir       = data_dir,
        label_norm     = label_norm,
        augment        = augment,
        s1_index_list  = configs.s1_index_list,
        s2_index_list  = configs.s2_index_list,
        months_list    = configs.months_list,
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        num_workers = configs.num_workers,
        pin_memory  = configs.pin_memory,
        drop_last   = drop_last,
        shuffle     = shuffle,
    )
    return dataloader