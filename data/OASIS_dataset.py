from torch.utils.data import Dataset
import scipy.io as sio
import data.util_3D as Util
import os
import random
import numpy as np
import nibabel as nib
import torch.nn.functional as F
class OASISDataset(Dataset):
    def __init__(self, dataroot, split='train', pairs_json="", train_max_index=None):        
        self.split = split
        self.dataroot = dataroot
        
        if self.split in ('val', 'test'):
            import json
            with open(pairs_json, 'r') as f:
                dataset_info = json.load(f)
            key = 'registration_val' if self.split == 'val' else 'registration_test'
            self.pair_list = dataset_info[key]
            self.data_len = len(self.pair_list)
        else:
            dataFiles = sorted([f for f in os.listdir(dataroot) if f.endswith('.nii.gz')])
            self.imageNum = [f for f in dataFiles if int(f.split('_')[1]) < train_max_index]
            self.data_len = len(self.imageNum)

    def __len__(self):
        return self.data_len

    def load_nii(self, path):
       # cargar y normalizar a [0, 1]
        img = nib.load(path).get_fdata()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return np.float32(img)

    def __getitem__(self, index):
        if self.split in ('val', 'test'):
            pair = self.pair_list[index]
            img_name_F = os.path.basename(pair['fixed'])
            img_name_M = os.path.basename(pair['moving'])

            path_img_F = os.path.join(self.dataroot, img_name_F)
            path_img_M = os.path.join(self.dataroot, img_name_M)
        else:
            img_name_M = self.imageNum[index]
            path_img_M = os.path.join(self.dataroot, img_name_M)

            index_F = random.randint(0, self.data_len - 1)
            img_name_F = self.imageNum[index_F]
            path_img_F = os.path.join(self.dataroot, img_name_F)

        # labels
        #path_lab_M = path_img_M.replace('imagesTr', 'labelsTr')
        #path_lab_F = path_img_F.replace('imagesTr', 'labelsTr')
#
        #dataM = self.load_nii(path_img_M)
        #labelM = nib.load(path_lab_M).get_fdata()
#
        #dataF = self.load_nii(path_img_F)
        #labelF = nib.load(path_lab_F).get_fdata()
#
#
        dataM = self.load_nii(path_img_M)
        dataF = self.load_nii(path_img_F)

        if self.split == 'test':
            # Ajusta 'labelsTs' si tu carpeta de etiquetas de test se llama así (o déjalo en 'labelsTr' si están en la misma)
            path_lab_M = path_img_M.replace('imagesTr', 'labelsTs').replace('imagesTs', 'labelsTs').replace('.nii.gz', '.mat')
            path_lab_F = path_img_F.replace('imagesTr', 'labelsTs').replace('imagesTs', 'labelsTs').replace('.nii.gz', '.mat')
            
            mat_M = sio.loadmat(path_lab_M)
            mat_F = sio.loadmat(path_lab_F)
            
            # Extrae la matriz ignorando las claves internas de MATLAB ('__header__', '__version__', '__globals__')
            key_M = [k for k in mat_M.keys() if not k.startswith('__')][0]
            key_F = [k for k in mat_F.keys() if not k.startswith('__')][0]
            
            labelM = mat_M[key_M].astype(np.float32)
            labelF = mat_F[key_F].astype(np.float32)
        else:
            path_lab_M = path_img_M.replace('imagesTr', 'labelsTr')
            path_lab_F = path_img_F.replace('imagesTr', 'labelsTr')
            labelM = nib.load(path_lab_M).get_fdata()
            labelF = nib.load(path_lab_F).get_fdata()

        
        [dataM, dataF, labelM, labelF] = Util.augment([dataM, dataF, labelM, labelF],  split=self.split)

        # convertir a tensores y a [-1, 1]
        dataM_t = Util.transform2tensor(Util.transform2numpy(dataM), min_max=(-1, 1))
        dataF_t = Util.transform2tensor(Util.transform2numpy(dataF), min_max=(-1, 1))
        
        # convertir a tensores y a [0, 1] para las labels
        labelM_t = Util.transform2tensor(Util.transform2numpy(labelM), min_max=(0, 1))
        labelF_t = Util.transform2tensor(Util.transform2numpy(labelF), min_max=(0, 1))

        # compresion 0.5
        
        # Imágenes médicas -> trilineal
        data = F.interpolate(dataM_t.unsqueeze(0), scale_factor=0.5, mode='trilinear', align_corners=False).squeeze(0)
        label = F.interpolate(dataF_t.unsqueeze(0), scale_factor=0.5, mode='trilinear', align_corners=False).squeeze(0)

        # labels -> nearest
        labelM = F.interpolate(labelM_t.unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0)
        labelF = F.interpolate(labelF_t.unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0)

        
        return {
            'M': data, 
            'F': label, 
            'MS': labelM, 
            'FS': labelF, 
            'nS': 7, 
            'P': path_img_M, 
            'Index': index
        }

