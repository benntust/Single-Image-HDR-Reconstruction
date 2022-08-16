import numpy as np
import cv2

def ssim(img1, img2):
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2

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
    img1, img2: [0, 1]
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


def mu_tonemap(hdr_image, mu=5000):
    
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)

def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    
    bounded_hdr = hdr_image / norm_value
    return  mu_tonemap(bounded_hdr, mu)

def psnr_norm_mu_tonemap(hdr_linear_res, hdr_linear_ref, percentile=99, gamma=2.24):
    
    '''hdr_linear_ref = hdr_linear_ref**gamma
    hdr_linear_res = hdr_linear_res**gamma'''
    norm = np.max(hdr_linear_ref)
    hdr_linear_res = norm_mu_tonemap(hdr_linear_res, norm) 
    hdr_linear_ref = norm_mu_tonemap(hdr_linear_ref, norm)
    #hdr_linear_res_clip = np.clip(hdr_linear_res, 0.0, 1.0)
    #hdr_linear_ref_clip = np.clip(hdr_linear_ref, 0.0, 1.0)

    return psnr(hdr_linear_res, hdr_linear_ref), calculate_ssim(hdr_linear_res, hdr_linear_ref)

def calculate_psnr(pre_hdr,gt_hdr):
    mse = np.mean(np.power(pre_hdr-gt_hdr, 2))
    if(mse == 0):
        print("mse=0 !!")
        return np.power(10,-6)
    max_pixel = 1.0 #gt_max
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    
    bounded_hdr = np.tanh(hdr_image / norm_value)
    return  mu_tonemap(bounded_hdr, mu)

def psnr_tanh_norm_mu_tonemap(hdr_linear_res, hdr_linear_ref, percentile=99, gamma=2.24):
    
    '''hdr_linear_ref = hdr_nonlinear_ref**gamma
    hdr_linear_res = hdr_nonlinear_res**gamma'''
    norm_perc = np.percentile(hdr_linear_ref, percentile)
    hdr_linear_res = tanh_norm_mu_tonemap(hdr_linear_res, norm_perc) 
    hdr_linear_ref = tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc)
    #hdr_linear_res_clip = np.clip(hdr_linear_res, 0.0, 1.0)
    #hdr_linear_ref_clip = np.clip(hdr_linear_ref, 0.0, 1.0)

    return psnr(hdr_linear_res, hdr_linear_ref), calculate_ssim(hdr_linear_res, hdr_linear_ref)

def psnr(im0, im1):
    
    return -10*np.log10(np.mean(np.power(im0-im1, 2)))

def normalized_psnr(im0, im1):
    norm = np.max([im0, im1])
    return psnr(im0/norm, im1/norm), calculate_ssim(im0/norm, im1/norm)

if __name__ == '__main__':
    g_hdr = cv2.imread("./results/inference/Pix2Pix_Results_001_target.png", -1)
    gt_hdr = cv2.imread("./results/inference/Pix2Pix_Results_001_after.png", -1)
    #gt_hdr = preprocessing(gt_hdr, False)
    #tonemapping
    '''tone_g_hdr = mu_tone(g_hdr)
    tone_gt_hdr = mu_tone(gt_hdr)'''
    #compute psnr and ssim
    psnr_result = psnr_tanh_norm_mu_tonemap(g_hdr,gt_hdr)
    ssim_result = psnr_tanh_norm_mu_tonemap(g_hdr,gt_hdr)
    print("PSNR=",psnr_result)
    print("SSIM=",ssim_result)
