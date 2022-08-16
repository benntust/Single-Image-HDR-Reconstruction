import os
import numpy as np
import cv2
import pickle
from numpy.core.fromnumeric import var
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CURR_PATH_PREFIX = os.path.dirname(os.path.abspath(__file__))

# get crf
def _get_crf_list():
    with open(os.path.join(CURR_PATH_PREFIX, 'dorfCurves.txt'), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    
    crf_list = [lines[idx + 5] for idx in range(0, len(lines), 6)] # get B curve
    #print(np.shape(crf_list)) # 201 lines
    crf_list = np.float32([ele.split() for ele in crf_list]) # (201,1024)
    np.random.RandomState(730).shuffle(crf_list)
    train_crf_list = crf_list[:-10]
    test_crf_list = crf_list[-10:] #10 crf for testing
    
    return train_crf_list, test_crf_list

train_crf_list, test_crf_list = _get_crf_list()

# get t_list
_get_t_list = lambda n: 2 ** np.linspace(-3, 3, n, dtype='float32')
train_t_list = _get_t_list(7)
test_t_list = _get_t_list(7) 

class HDRDataset(Dataset):

    def __init__(self, hdr_list, crf_list, t_list, is_train):
        self.hdr_list = hdr_list
        self.crf_list = crf_list
        self.t_list = t_list
        self.is_train = is_train

    def __getitem__(self, idx):
        #get each idx
        hdr_idx = (idx // len(self.t_list)) % len(self.hdr_list)
        #crf_idx = np.random.randint(len(self.crf_list))
        #crf_idx = idx % len(self.crf_list)
        t_idx = idx % len(self.t_list)
        # read hdr and resize
        #hdr_path = os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_train"), self.hdr_list[hdr_idx])
        if self.is_train:
            hdr_path = os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_train"), self.hdr_list[hdr_idx])
        else:
            hdr_path = os.path.join(os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_test"), self.hdr_list[hdr_idx]), "gt.hdr")
        #print(hdr_path)
        hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        if(idx >= self.__len__()/2):
            left = True
        else:
            left = False
        hdr = self.preprocessing(hdr, left)
        if self.is_train:
            hdr = self.transform(hdr)
        #crf = self.crf_list[crf_idx]
        t = self.t_list[t_idx]
        t_target = t/4
        ldr_22 = self.hdr2ldr(hdr, t)
        ldr_target_22 = self.hdr2ldr(hdr, t_target)
        '''if self.is_train:
            ldr_22, ldr_target_22 = self.ldr_transform(ldr_22, ldr_target_22)'''
        # use .copy() to slove the error
        # ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported.
        ldr_22 = transforms.functional.to_tensor(ldr_22.copy())
        ldr_target_22 = transforms.functional.to_tensor(ldr_target_22.copy())
        # normailize
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ldr_22 = normalize(ldr_22)
        ldr_target_22 = normalize(ldr_target_22)
        #print(ldr1) -1 to 1
        return ldr_22, ldr_target_22 #hdr, ldr, crf, t

    def __len__(self):
        if self.is_train:
            return 2* len(self.hdr_list)* len(self.t_list)
        else:
            return len(self.hdr_list)* len(self.t_list)

    def preprocessing(self, hdr, left):
        # BGR to RGB and non-neg
        hdr = hdr[:,:,::-1]
        hdr = np.clip(hdr, 0, None)
        # resize
        h, w, _, = hdr.shape
        ratio = np.max([512 / h, 512 / w])
        h = int(np.round(h * ratio))
        w = int(np.round(w * ratio))
        hdr = cv2.resize(hdr, (w, h), cv2.INTER_AREA)
        # cut hdr to left or right part
        if h > w:
            hdr = hdr[:512, :, :] if left else hdr[-512:, :, :]
        else:
            hdr = hdr[:, :512, :] if left else hdr[:, -512:, :]
        # make mean to 0.5
        hdr_mean = np.mean(hdr)
        hdr = 0.5 * hdr / (hdr_mean + 1e-6)

        return hdr

    def transform(self, hdr):
        # rescale
        scale = np.random.uniform(1.0, 2.0)
        hdr = cv2.resize(hdr, (np.round(512 * scale).astype(np.int32), np.round(512 * scale).astype(np.int32)), cv2.INTER_AREA)
        # random crop
        def randomCrop(img, width, height):
            assert img.shape[0] >= height
            assert img.shape[1] >= width
            if(img.shape[0] == height and img.shape[1] == width):
                return img
            x = np.random.randint(0, img.shape[1] - width)
            y = np.random.randint(0, img.shape[0] - height)
            img = img[y:y + height, x:x + width]
            return img
        hdr = randomCrop(hdr, 512, 512)
        # random rotation
        hdr = np.rot90(hdr, np.random.randint(4))
        # random flip
        if np.random.choice([True, False]):
            hdr = np.flip(hdr, 0)
        if np.random.choice([True, False]):
            hdr = np.flip(hdr, 1)
        return hdr

    def hdr2ldr(self, hdr, t):
        hdr = hdr * t
        hdr = np.clip(hdr, 0, 1)

        ldr_22 = np.power(hdr,1/2.2)
        ldr_22 = np.round(ldr_22 * 255.0)
        ldr_22 = np.clip(ldr_22, 0, 255).astype(np.uint8)

        return ldr_22 #ldr ,ldr_22
    
    def ldr_transform(self,ldr1,ldr2):
        # gamma correction
        if(np.random.uniform(0,1) <= 0.4):
            gamma = np.round(np.random.uniform(0.6, 1.4),2)
            table = [((i / 255) ** gamma) * 255 for i in range(256)]
            table = np.array(table, np.uint8)
            ldr1 = cv2.LUT(ldr1, table)
            ldr2 = cv2.LUT(ldr2, table)
        # saturate change
        if(np.random.uniform(0,1) <= 0.4):
            weight = np.round(np.random.uniform(0.8, 1.2),2)
            ldr1_gray = cv2.cvtColor(ldr1, cv2.COLOR_RGB2GRAY)
            ldr2_gray = cv2.cvtColor(ldr2, cv2.COLOR_RGB2GRAY)
            # expand to 3 channel
            ldr1_gray = cv2.cvtColor(ldr1_gray, cv2.COLOR_GRAY2RGB)
            ldr2_gray = cv2.cvtColor(ldr2_gray, cv2.COLOR_GRAY2RGB)
            ldr1 = np.clip(weight * ldr1 + (1-weight) * ldr1_gray, 0, 255).astype(np.uint8)
            ldr2 = np.clip(weight * ldr2 + (1-weight) * ldr2_gray, 0, 255).astype(np.uint8)
        # brightness change
        if(np.random.uniform(0,1) <= 0.4):
            weight = np.round(np.random.uniform(0.8, 1.2),2)
            ldr1_hsv = cv2.cvtColor(ldr1, cv2.COLOR_RGB2HSV)
            ldr2_hsv = cv2.cvtColor(ldr2, cv2.COLOR_RGB2HSV)
            ldr1_hsv[:,:,2] = np.clip(ldr1_hsv[:,:,2] * weight, 0, 255).astype(np.uint8)
            ldr2_hsv[:,:,2] = np.clip(ldr2_hsv[:,:,2] * weight, 0, 255).astype(np.uint8)
            ldr1 = cv2.cvtColor(ldr1_hsv, cv2.COLOR_HSV2RGB)
            ldr2 = cv2.cvtColor(ldr2_hsv, cv2.COLOR_HSV2RGB)
        return ldr1, ldr2

def _load_pkl(name):
    with open(os.path.join(CURR_PATH_PREFIX, name + '.pkl'), 'rb') as f:
        out = pickle.load(f)
    return out

# get_train_dataset
def get_train_dataset():
    i_dataset_train_posfix_list = _load_pkl('i_dataset_train')
    return HDRDataset(
        i_dataset_train_posfix_list,
        train_crf_list,
        train_t_list,
        True
    )

# get_vali_dataset
def get_vali_dataset():
    # get validation set (from test set)
    i_dataset_vali_posfix_list = os.listdir(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_test"))
    #i_dataset_vali_posfix_list = _load_pkl('i_dataset_train')

    vali_crf_list = test_crf_list.copy() 

    return HDRDataset(
        i_dataset_vali_posfix_list,
        vali_crf_list,
        test_t_list,
        False
    )