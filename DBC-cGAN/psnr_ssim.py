import numpy as np
import cv2

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def mu_tone(hdr):
    mu = 5000
    tone_hdr = np.log(1+mu*hdr)/np.log(1+mu)
    return tone_hdr

def calculate_psnr(pre_hdr,gt_hdr):
    mse = np.mean(np.power(pre_hdr-gt_hdr, 2))
    if(mse == 0):
        print("mse=0 !!")
        return np.power(10,-6)
    max_pixel = 1 #gt_max
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def preprocessing(hdr, left):
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

if __name__ == '__main__':
    g_hdr = cv2.imread("./results/inference/Pix2Pix_Results_001_target.png", -1)
    gt_hdr = cv2.imread("./results/inference/Pix2Pix_Results_001_after.png", -1)
    #gt_hdr = preprocessing(gt_hdr, False)
    #tonemapping
    '''tone_g_hdr = mu_tone(g_hdr)
    tone_gt_hdr = mu_tone(gt_hdr)'''
    #compute psnr and ssim
    psnr_result = calculate_psnr(g_hdr,gt_hdr)
    ssim_result = calculate_ssim(g_hdr,gt_hdr)
    print("PSNR=",psnr_result)
    print("SSIM=",ssim_result)
