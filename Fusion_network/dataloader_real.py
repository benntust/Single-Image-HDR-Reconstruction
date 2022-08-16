import os
import random
import numpy as np
import cv2
import pickle
from numpy.core.fromnumeric import var
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CURR_PATH_PREFIX = os.path.dirname(os.path.abspath(__file__))

class HDRDataset(Dataset):

    def __init__(self, hdr_list, is_train):
        self.hdr_list = hdr_list
        self.is_train = is_train

    def __getitem__(self, idx):
        #get each idx
        hdr_idx = idx % len(self.hdr_list)
        #crf_idx = idx % len(self.crf_list)
        # read hdr and resize
        #hdr_path = os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_train"), self.hdr_list[hdr_idx])
        if self.is_train:
            hdr_path = os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Real_train/HDR_gt"), self.hdr_list[hdr_idx])
            ldr_path = os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Real_train/LDR_in"), self.hdr_list[hdr_idx].replace("hdr","jpg"))
        else:
            hdr_path = os.path.join(os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Real_test"), self.hdr_list[hdr_idx]), "gt.hdr")
            ldr_path = os.path.join(os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Real_test"), self.hdr_list[hdr_idx]), "input.jpg")
        #print(hdr_path)
        hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        hdr = self.preprocessing(hdr)
        ldr = cv2.imread(ldr_path, cv2.IMREAD_UNCHANGED)
        ldr = ldr[:,:,::-1]
        if self.is_train:
            hdr,ldr = self.transform(hdr,ldr)
            ldr= self.ldr_transform(ldr)
        hdr = transforms.functional.to_tensor(hdr.copy())
        ldr = transforms.functional.to_tensor(ldr.copy())
        # normailize
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ldr = normalize(ldr)
        #print(ldr1) -1 to 1
        return ldr, hdr

    def __len__(self):
        return len(self.hdr_list)#* len(self.t_list) #8400

    def preprocessing(self, hdr):
        # BGR to RGB and non-neg
        hdr = hdr[:,:,::-1]
        hdr = np.clip(hdr, 0, None)
        # make mean to 0.5           
        hdr_mean = np.mean(hdr)
        hdr = 0.5 * hdr / (hdr_mean + 1e-6)
        return hdr

    def transform(self, hdr, ldr):
        # rescale
        scale = np.random.uniform(1.0, 2.0)
        hdr = cv2.resize(hdr, (np.round(512 * scale).astype(np.int32), np.round(512 * scale).astype(np.int32)), cv2.INTER_AREA)
        ldr = cv2.resize(ldr, (np.round(512 * scale).astype(np.int32), np.round(512 * scale).astype(np.int32)), cv2.INTER_AREA)
        # random crop
        def randomCrop(hdr, ldr, width, height):
            assert hdr.shape[0] >= height
            assert hdr.shape[1] >= width
            assert hdr.shape[0] == ldr.shape[0]
            assert hdr.shape[1] == ldr.shape[1]
            if(hdr.shape[0] == height and hdr.shape[1] == width):
                return hdr, ldr
            x = np.random.randint(0, hdr.shape[1] - width)
            y = np.random.randint(0, hdr.shape[0] - height)
            hdr = hdr[y:y + height, x:x + width]
            ldr = ldr[y:y + height, x:x + width]
            return hdr, ldr
        hdr, ldr = randomCrop(hdr, ldr, 512, 512)
        # random rotation
        direction = np.random.randint(4)
        hdr = np.rot90(hdr, direction)
        ldr = np.rot90(ldr, direction)
        # random flip
        if np.random.choice([True, False]):
            hdr = np.flip(hdr, 0)
            ldr = np.flip(ldr, 0)
        if np.random.choice([True, False]):
            hdr = np.flip(hdr, 1)
            ldr = np.flip(ldr, 1)
        return hdr, ldr

    def hdr2ldr(self, hdr, t):
        
        hdr = hdr * t
        hdr = np.clip(hdr, 0, 1)

        ldr_22 = np.power(hdr,1/2.2)
        ldr_22 = np.round(ldr_22 * 255.0)
        ldr_22 = np.clip(ldr_22, 0, 255).astype(np.uint8)

        return ldr_22 #ldr ,ldr_22

    def ldr_transform(self,ldr1):
        # gamma correction
        if(np.random.uniform(0,1) <= 0.5):
            gamma = np.round(np.random.uniform(0.6, 1.4),2)
            table = [((i / 255) ** gamma) * 255 for i in range(256)]
            table = np.array(table, np.uint8)
            ldr1 = cv2.LUT(ldr1, table)
        # saturate change
        if(np.random.uniform(0,1) <= 0.5):
            weight = np.round(np.random.uniform(0.8, 1.2),2)
            ldr1_gray = cv2.cvtColor(ldr1, cv2.COLOR_RGB2GRAY)
            # expand to 3 channel
            ldr1_gray = cv2.cvtColor(ldr1_gray, cv2.COLOR_GRAY2RGB)
            ldr1 = np.clip(weight * ldr1 + (1-weight) * ldr1_gray, 0, 255).astype(np.uint8)
        # brightness change
        if(np.random.uniform(0,1) <= 0.5):
            weight = np.round(np.random.uniform(0.8, 1.2),2)
            ldr1_hsv = cv2.cvtColor(ldr1, cv2.COLOR_RGB2HSV)
            ldr1_hsv[:,:,2] = np.clip(ldr1_hsv[:,:,2] * weight, 0, 255).astype(np.uint8)
            ldr1 = cv2.cvtColor(ldr1_hsv, cv2.COLOR_HSV2RGB)
        return ldr1
    
def _load_pkl(name):
    with open(os.path.join(CURR_PATH_PREFIX, name + '.pkl'), 'rb') as f:
        out = pickle.load(f)
    return out

# get_train_dataset
def get_train_dataset():
    i_dataset_train_posfix_list = os.listdir(os.path.join(CURR_PATH_PREFIX, "HDR-Real_train/HDR_gt"))
    return HDRDataset(
        i_dataset_train_posfix_list,
        True
    )

# get_vali_dataset
def get_vali_dataset():
    # get validation set (from test set)
    i_dataset_vali_posfix_list = os.listdir(os.path.join(CURR_PATH_PREFIX, "HDR-Real_test"))
    i_dataset_vali_posfix_list = sorted(i_dataset_vali_posfix_list)

    return HDRDataset(
        i_dataset_vali_posfix_list,
        False
    )

# get_test_dataset
def get_test_dataset():
    i_dataset_test_posfix_list = _load_pkl('i_dataset_test')
    return HDRDataset(
        i_dataset_test_posfix_list,
        False
    )

def get_inference_dataset():
    i_dataset_test_posfix_list = os.listdir(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_test"))
    return HDRDataset(
        i_dataset_test_posfix_list,
        False
    )

if __name__ == "__main__":
    get_vali_dataset()