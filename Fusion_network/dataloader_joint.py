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
    
    '''plt.figure()
    for crf in train_crf_list:
        plt.plot(crf)
    plt.savefig("crf.png")
    plt.show()'''
    
    return train_crf_list, test_crf_list

train_crf_list, test_crf_list = _get_crf_list()

# --- t_list
_get_t_list = lambda n: 2 ** np.linspace(-3, 3, n, dtype='float32')
t_list = _get_t_list(7) # darker

class HDRDataset(Dataset):

    def __init__(self, hdr_list, crf_list, t_list, is_train):
        self.hdr_list = hdr_list
        self.crf_list = crf_list
        self.t_list = t_list
        self.is_train = is_train

    def __getitem__(self, idx):
        #get each idx
        hdr_idx = idx % len(self.hdr_list)
        crf_idx = np.random.randint(len(self.crf_list))
        t_idx = (idx // len(self.hdr_list)) % len(self.t_list)
        # read hdr and resize
        #hdr_path = os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_train"), self.hdr_list[hdr_idx])
        if self.is_train:
            hdr_path = os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_train"), self.hdr_list[hdr_idx])
        else:
            hdr_path = os.path.join(os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_test"), self.hdr_list[hdr_idx]), "gt.hdr")
            ldr_path = os.path.join(os.path.join(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_test"), self.hdr_list[hdr_idx]), "input.jpg")
        #print(hdr_path)
        hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        if(idx >= self.__len__()/2):
            left = True
        else:
            left = False
        hdr = self.preprocessing(hdr, left)
        if self.is_train:
            hdr = self.transform(hdr)
            crf = self.crf_list[crf_idx]
            t = self.t_list[t_idx]
            b1 = t*16
            b2 = t*4
            d1 = t/4
            d2 = t/16
            ldr = self.hdr2ldr(hdr, crf, t)
            ldr_b1 = self.hdr2ldr(hdr, crf, b1)
            ldr_b2 = self.hdr2ldr(hdr, crf, b2)
            ldr_d1 = self.hdr2ldr(hdr, crf, d1)
            ldr_d2 = self.hdr2ldr(hdr, crf, d2)
            #ldr, ldr_b2, ldr_b2, ldr_d1, ldr_d2 = self.ldr_transform(ldr, ldr_b2, ldr_b2, ldr_d1, ldr_d2)
        else:
            ldr = cv2.imread(ldr_path, cv2.IMREAD_UNCHANGED)
            ldr = ldr[:,:,::-1]
            t = self.t_list[(idx//1200)]
            '''crf = self.crf_list[crf_idx]
            t = self.t_list[t_idx]
            ldr = self.hdr2ldr(hdr, crf, t)  '''   
        # use .copy() to slove the error
        # ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported.
        # hdr = torch.from_numpy(hdr.copy())
        if self.is_train:
            hdr = transforms.functional.to_tensor(hdr.copy())
            ldr = transforms.functional.to_tensor(ldr.copy())
            ldr_b1 = transforms.functional.to_tensor(ldr_b1.copy())
            ldr_b2 = transforms.functional.to_tensor(ldr_b2.copy())
            ldr_d1 = transforms.functional.to_tensor(ldr_d1.copy())
            ldr_d2 = transforms.functional.to_tensor(ldr_d2.copy())
            # normailize
            normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ldr = normalize(ldr)
            ldr_b1 = normalize(ldr_b1)
            ldr_b2 = normalize(ldr_b2)
            ldr_d1 = normalize(ldr_d1)
            ldr_d2 = normalize(ldr_d2)
            #print(ldr1) -1 to 1
            return ldr, ldr_b1, ldr_b2, ldr_d1, ldr_d2, hdr
        else:
            hdr = transforms.functional.to_tensor(hdr.copy())
            ldr = transforms.functional.to_tensor(ldr.copy())
            # normailize
            normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ldr = normalize(ldr)
            #print(ldr1) -1 to 1
            return ldr, hdr, t #hdr, ldr, crf, t

    def __len__(self):
        if self.is_train:
            return 2* len(self.hdr_list)* len(self.t_list)
        else:
            return len(self.hdr_list)#* len(self.t_list) #8400

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

    def hdr2ldr(self, hdr, crf, t):
        ldr_22 = np.power(hdr,1/2.2)
        ldr_22 = np.round(ldr_22 * 255.0)
        ldr_22 = np.clip(ldr_22, 0, 255).astype(np.uint8)

        return ldr_22 #ldr ,ldr_22

    def ldr_transform(self,ldr1,ldr2,ldr3,ldr4,ldr5):
        # gamma correction
        if(np.random.uniform(0,1) <= 0.5):
            gamma = np.round(np.random.uniform(0.6, 1.4),2)
            table = [((i / 255) ** gamma) * 255 for i in range(256)]
            table = np.array(table, np.uint8)
            ldr1 = cv2.LUT(ldr1, table)
            ldr2 = cv2.LUT(ldr2, table)
            ldr3 = cv2.LUT(ldr3, table)
            ldr4 = cv2.LUT(ldr4, table)
            ldr5 = cv2.LUT(ldr5, table)
        # saturate change
        if(np.random.uniform(0,1) <= 0.5):
            weight = np.round(np.random.uniform(0.8, 1.2),2)
            ldr1_gray = cv2.cvtColor(ldr1, cv2.COLOR_RGB2GRAY)
            ldr2_gray = cv2.cvtColor(ldr2, cv2.COLOR_RGB2GRAY)
            ldr3_gray = cv2.cvtColor(ldr3, cv2.COLOR_RGB2GRAY)
            ldr4_gray = cv2.cvtColor(ldr4, cv2.COLOR_RGB2GRAY)
            ldr5_gray = cv2.cvtColor(ldr5, cv2.COLOR_RGB2GRAY)
            # expand to 3 channel
            ldr1_gray = cv2.cvtColor(ldr1_gray, cv2.COLOR_GRAY2RGB)
            ldr2_gray = cv2.cvtColor(ldr2_gray, cv2.COLOR_GRAY2RGB)
            ldr3_gray = cv2.cvtColor(ldr3_gray, cv2.COLOR_GRAY2RGB)
            ldr4_gray = cv2.cvtColor(ldr4_gray, cv2.COLOR_GRAY2RGB)
            ldr5_gray = cv2.cvtColor(ldr5_gray, cv2.COLOR_GRAY2RGB)
            ldr1 = np.clip(weight * ldr1 + (1-weight) * ldr1_gray, 0, 255).astype(np.uint8)
            ldr2 = np.clip(weight * ldr2 + (1-weight) * ldr2_gray, 0, 255).astype(np.uint8)
            ldr3 = np.clip(weight * ldr3 + (1-weight) * ldr3_gray, 0, 255).astype(np.uint8)
            ldr4 = np.clip(weight * ldr4 + (1-weight) * ldr4_gray, 0, 255).astype(np.uint8)
            ldr5 = np.clip(weight * ldr5 + (1-weight) * ldr5_gray, 0, 255).astype(np.uint8)
        # brightness change
        if(np.random.uniform(0,1) <= 0.5):
            weight = np.round(np.random.uniform(0.8, 1.2),2)
            ldr1_hsv = cv2.cvtColor(ldr1, cv2.COLOR_RGB2HSV)
            ldr2_hsv = cv2.cvtColor(ldr2, cv2.COLOR_RGB2HSV)
            ldr3_hsv = cv2.cvtColor(ldr3, cv2.COLOR_RGB2HSV)
            ldr4_hsv = cv2.cvtColor(ldr4, cv2.COLOR_RGB2HSV)
            ldr5_hsv = cv2.cvtColor(ldr5, cv2.COLOR_RGB2HSV)
            ldr1_hsv[:,:,2] = np.clip(ldr1_hsv[:,:,2] * weight, 0, 255).astype(np.uint8)
            ldr2_hsv[:,:,2] = np.clip(ldr2_hsv[:,:,2] * weight, 0, 255).astype(np.uint8)
            ldr3_hsv[:,:,2] = np.clip(ldr3_hsv[:,:,2] * weight, 0, 255).astype(np.uint8)
            ldr4_hsv[:,:,2] = np.clip(ldr4_hsv[:,:,2] * weight, 0, 255).astype(np.uint8)
            ldr5_hsv[:,:,2] = np.clip(ldr5_hsv[:,:,2] * weight, 0, 255).astype(np.uint8)
            ldr1 = cv2.cvtColor(ldr1_hsv, cv2.COLOR_HSV2RGB)
            ldr2 = cv2.cvtColor(ldr2_hsv, cv2.COLOR_HSV2RGB)
            ldr3 = cv2.cvtColor(ldr3_hsv, cv2.COLOR_HSV2RGB)
            ldr4 = cv2.cvtColor(ldr4_hsv, cv2.COLOR_HSV2RGB)
            ldr5 = cv2.cvtColor(ldr5_hsv, cv2.COLOR_HSV2RGB)
        return ldr1, ldr2, ldr3, ldr4, ldr5
    
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
        t_list,
        True
    )

# get_vali_dataset
def get_vali_dataset():
    # get validation set (from test set)
    i_dataset_vali_posfix_list = os.listdir(os.path.join(CURR_PATH_PREFIX, "HDR-Synth_test"))
    i_dataset_vali_posfix_list = sorted(i_dataset_vali_posfix_list)
    #print(i_dataset_vali_posfix_list)
    i_dataset_vali_posfix_list_one = []
    for idx in range(0,len(i_dataset_vali_posfix_list),1200):
        for add in range(0,120):
            ran = random.randint(0,9)
            i_dataset_vali_posfix_list_one.append(i_dataset_vali_posfix_list[idx+ran*120+add])
    #i_dataset_vali_posfix_list = _load_pkl('i_dataset_train')'''
    #print(len(i_dataset_vali_posfix_list))

    vali_crf_list = test_crf_list.copy() 
    #print(i_dataset_vali_posfix_list_one)

    return HDRDataset(
        i_dataset_vali_posfix_list_one,
        vali_crf_list,
        t_list,
        False
    )

if __name__ == "__main__":
    get_vali_dataset()