import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from os.path import join as opj

from ..process import *
from ..models import define_model
from torchsummary import summary

class BHEPredictor(object):

    def __init__(self, model_path, configs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_dict = torch.load(model_path, map_location='cpu')
        model = define_model(configs.model)
        model.load_state_dict(model_dict)
        model = model.to(self.device)
        model.eval()
        self.model = model

        self.months_list = configs.loader.months_list
        if configs.loader.months_list == 'all':
            self.months_list = list(range(12))

        self.s1_index_list = configs.loader.s1_index_list
        if configs.loader.s1_index_list == 'all':
            self.s1_index_list = list(range(4))
        
        self.s2_index_list = configs.loader.s2_index_list
        if configs.loader.s2_index_list == 'all':
            self.s2_index_list = list(range(11))

    @torch.no_grad()
    def predict(self, data_dir, subjects, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for subject in tqdm(subjects, ncols=88):
            feature, mask = self._load_data(data_dir, subject)
            feature = torch.from_numpy(feature)
            feature = feature.to(self.device)        
            mask = torch.from_numpy(mask)
            mask = mask.to(self.device)

            pred, rseg, seg = self.model(feature, mask)           
            pred = pred.cpu().numpy()[0, 0]
            seg = seg.squeeze().cpu().numpy()
            pred = Image.fromarray(pred)
            output_path = opj(output_dir, f'{subject[:-4]}_pred.tif')
            pred.save(output_path, format='TIFF', save_all=True)
            
            seg = Image.fromarray(seg[:, :])           
            output_path = opj(output_dir, f'{subject[:-4]}_seg.tif')
            seg.save(output_path, format='TIFF', save_all=True)
    
    def _load_data(self, data_dir, subject):
        # loads S1 and S2 features
        feature_list, mask = [], []
        for month in self.months_list:
            file_name = '%s_%02d.tif' % (str.split(subject, '.')[0], month)
            s1_path = opj(data_dir, 'S1', file_name)
            s2_path = opj(data_dir, 'S2', file_name)
            
            img_s1 = GRD_toRGB_S1(s1_path)
            s1_list = img_s1.astype("float32")
            img_s2 = GRD_toRGB_S2(s2_path)
            s2_list = img_s2.astype("float32")

            feature = np.concatenate([s1_list , s2_list], axis=-1)
            feature = np.expand_dims(feature, axis=0)
            feature_list.append(feature)
            mask.append(False)
        feature = np.concatenate(feature_list, axis=0)
        feature = feature.transpose(0, 3, 1, 2).astype(np.float32)
        feature = np.expand_dims(feature, axis=0)
        
        mask = np.array(mask)
        
        return feature, mask