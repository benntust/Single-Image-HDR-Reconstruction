import numpy as np
import cv2
import os
import shutil

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
    ssims = []
    for i in range(3):
        ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
    return np.array(ssims).mean()

def mu_tonemap(hdr_image, mu=5000):
    
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)

def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    
    bounded_hdr = hdr_image / norm_value
    return  mu_tonemap(bounded_hdr, mu)

def psnr(im0, im1):
    
    return -10*np.log10(np.mean(np.power(im0-im1, 2)))

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

def normalized_psnr(im0, im1):
    norm = np.max([im0, im1])
    return psnr(im0/norm, im1/norm), calculate_ssim(im0/norm, im1/norm)

def preprocessing(hdr):
    # BGR to RGB and non-neg
    #hdr = hdr[:,:,::-1]
    hdr = np.clip(hdr, 0, None)
    # make mean to 0.5
    hdr_mean = np.mean(hdr)
    hdr = 0.5 * hdr / (hdr_mean + 1e-6)
    return hdr

def thres(gray, threshold, smaller):
    threshold = np.ones(np.shape(gray))*threshold
    if(smaller):
        res = (gray<=threshold).astype(np.float32)
    else:
        res = (gray>=threshold).astype(np.float32)
    pix = np.sum(res)
    return pix

images = os.listdir(r"C:\Users\user\Desktop\paper\thesis figures\HDR-Real_test\HDR-Real_test")
images = sorted(images)

hdr_drtmo_psnr_list=[]
hdr_drtmo_ssim_list=[]
hdr_expand_psnr_list=[]
hdr_expand_ssim_list=[]
hdr_hdrcnn_psnr_list=[]
hdr_hdrcnn_ssim_list=[]
hdr_hpeo_psnr_list=[]
hdr_hpeo_ssim_list=[]
hdr_ours_psnr_list=[]
hdr_ours_ssim_list=[]
hdr_ben_psnr_list=[]
hdr_ben_ssim_list=[]

norm_drtmo_psnr_list=[]
norm_drtmo_ssim_list=[]
norm_expand_psnr_list=[]
norm_expand_ssim_list=[]
norm_hdrcnn_psnr_list=[]
norm_hdrcnn_ssim_list=[]
norm_hpeo_psnr_list=[]
norm_hpeo_ssim_list=[]
norm_ours_psnr_list=[]
norm_ours_ssim_list=[]
norm_ben_psnr_list=[]
norm_ben_ssim_list=[]

d_count=0
b_count=0
print(len(images))
for i, (image) in enumerate(images):
    path = os.path.join(r"C:\Users\user\Desktop\paper\thesis figures\HDR-Real_test\HDR-Real_test",image)
    input = cv2.imread(os.path.join(path,"input.jpg"), -1)
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    gray = gray.flatten()
    over_exposed_pix = thres(gray, 249, False)
    under_exposed_pix = thres(gray, 6, True)
    d_loss_const = under_exposed_pix>=(len(gray)*0.25)
    b_loss_const = over_exposed_pix>=(len(gray)*0.25)
    if(d_loss_const):
        d_count+=1
        continue
    if(b_loss_const):
        b_count+=1
        continue

    #shutil.copyfile(os.path.join(path,"ben.hdr"),os.path.join("/home/m10902159/project/THESIS/fusion/myfusion/ben/%d" %(i//250),"%d_ben.hdr" %(i)))
    #shutil.copyfile(os.path.join(path,"gt.hdr"),os.path.join("/home/m10902159/project/THESIS/fusion/myfusion/ben/%d" %(i//250),"%d_gt.hdr" %(i)))
    hdr_ben = cv2.imread(os.path.join(path,"ben.hdr"), -1) 
    hdr_gt = cv2.imread(os.path.join(path,"gt.hdr"), -1) 
    hdr_gt = preprocessing(hdr_gt)

    #compute psnr and ssim
    hdr_ben_psnr_result, hdr_ben_ssim_result = psnr_tanh_norm_mu_tonemap(hdr_ben,hdr_gt)
    hdr_ben_psnr_list.append(hdr_ben_psnr_result)
    hdr_ben_ssim_list.append(hdr_ben_ssim_result)

    norm_ben_psnr_result, norm_ben_ssim_result = normalized_psnr(hdr_ben,hdr_gt)
    
    norm_ben_psnr_list.append(norm_ben_psnr_result)
    norm_ben_ssim_list.append(norm_ben_ssim_result)

    hdr_drtmo = cv2.imread(os.path.join(path,"drtmo.hdr"), -1)
    hdr_expand = cv2.imread(os.path.join(path,"expand.hdr"), -1)
    hdr_hdrcnn = cv2.imread(os.path.join(path,"hdrcnn.hdr"), -1)
    hdr_hpeo = cv2.imread(os.path.join(path,"hpeo.hdr"), -1)
    hdr_ours = cv2.imread(os.path.join(path,"ours.hdr"), -1)

    #compute psnr and ssim
    hdr_drtmo_psnr_result, hdr_drtmo_ssim_result = psnr_tanh_norm_mu_tonemap(hdr_drtmo,hdr_gt)
    hdr_expand_psnr_result, hdr_expand_ssim_result = psnr_tanh_norm_mu_tonemap(hdr_expand,hdr_gt)
    hdr_hdrcnn_psnr_result, hdr_hdrcnn_ssim_result = psnr_tanh_norm_mu_tonemap(hdr_hdrcnn,hdr_gt)
    hdr_hpeo_psnr_result, hdr_hpeo_ssim_result = psnr_tanh_norm_mu_tonemap(hdr_hpeo,hdr_gt)
    hdr_ours_psnr_result, hdr_ours_ssim_result = psnr_tanh_norm_mu_tonemap(hdr_ours,hdr_gt)
    
    hdr_drtmo_psnr_list.append(hdr_drtmo_psnr_result)
    hdr_drtmo_ssim_list.append(hdr_drtmo_ssim_result)
    hdr_expand_psnr_list.append(hdr_expand_psnr_result)
    hdr_expand_ssim_list.append(hdr_expand_ssim_result)
    hdr_hdrcnn_psnr_list.append(hdr_hdrcnn_psnr_result)
    hdr_hdrcnn_ssim_list.append(hdr_hdrcnn_ssim_result)
    hdr_hpeo_psnr_list.append(hdr_hpeo_psnr_result)
    hdr_hpeo_ssim_list.append(hdr_hpeo_ssim_result)
    hdr_ours_psnr_list.append(hdr_ours_psnr_result)
    hdr_ours_ssim_list.append(hdr_ours_ssim_result)

    norm_drtmo_psnr_result, norm_drtmo_ssim_result = normalized_psnr(hdr_drtmo,hdr_gt)
    norm_expand_psnr_result, norm_expand_ssim_result = normalized_psnr(hdr_expand,hdr_gt)
    norm_hdrcnn_psnr_result, norm_hdrcnn_ssim_result = normalized_psnr(hdr_hdrcnn,hdr_gt)
    norm_hpeo_psnr_result, norm_hpeo_ssim_result = normalized_psnr(hdr_hpeo,hdr_gt)
    norm_ours_psnr_result, norm_ours_ssim_result = normalized_psnr(hdr_ours,hdr_gt)
    
    norm_drtmo_psnr_list.append(norm_drtmo_psnr_result)
    norm_drtmo_ssim_list.append(norm_drtmo_ssim_result)
    norm_expand_psnr_list.append(norm_expand_psnr_result)
    norm_expand_ssim_list.append(norm_expand_ssim_result)
    norm_hdrcnn_psnr_list.append(norm_hdrcnn_psnr_result)
    norm_hdrcnn_ssim_list.append(norm_hdrcnn_ssim_result)
    norm_hpeo_psnr_list.append(norm_hpeo_psnr_result)
    norm_hpeo_ssim_list.append(norm_hpeo_ssim_result)
    norm_ours_psnr_list.append(norm_ours_psnr_result)
    norm_ours_ssim_list.append(norm_ours_ssim_result)

print(d_count,"images under exposure")
print(b_count,"images over exposure")

print("hdr_drtmo_PSNR=",np.mean(hdr_drtmo_psnr_list))
print("hdr_drtmo_SSIM=",np.mean(hdr_drtmo_ssim_list))
print("hdr_expand_PSNR=",np.mean(hdr_expand_psnr_list))
print("hdr_expand_SSIM=",np.mean(hdr_expand_ssim_list))
print("hdr_hdrcnn_PSNR=",np.mean(hdr_hdrcnn_psnr_list))
print("hdr_hdrcnn_SSIM=",np.mean(hdr_hdrcnn_ssim_list))
print("hdr_hpeo_PSNR=",np.mean(hdr_hpeo_psnr_list))
print("hdr_hpeo_SSIM=",np.mean(hdr_hpeo_ssim_list))
print("hdr_ours_PSNR=",np.mean(hdr_ours_psnr_list))
print("hdr_ours_SSIM=",np.mean(hdr_ours_ssim_list))
print("hdr_ours_PSNR=",np.mean(hdr_ben_psnr_list))
print("hdr_ours_SSIM=",np.mean(hdr_ben_ssim_list))

print("\n\nnorm_drtmo_PSNR=",np.mean(norm_drtmo_psnr_list))
print("norm_drtmo_SSIM=",np.mean(norm_drtmo_ssim_list))
print("norm_expand_PSNR=",np.mean(norm_expand_psnr_list))
print("norm_expand_SSIM=",np.mean(norm_expand_ssim_list))
print("norm_hdrcnn_PSNR=",np.mean(norm_hdrcnn_psnr_list))
print("norm_hdrcnn_SSIM=",np.mean(norm_hdrcnn_ssim_list))
print("norm_hpeo_PSNR=",np.mean(norm_hpeo_psnr_list))
print("norm_hpeo_SSIM=",np.mean(norm_hpeo_ssim_list))
print("norm_ours_PSNR=",np.mean(norm_ours_psnr_list))
print("norm_ours_SSIM=",np.mean(norm_ours_ssim_list))
print("norm_hdr_ours_PSNR=",np.mean(norm_ben_psnr_list))
print("norm_hdr_ours_PSNR=",np.mean(norm_ben_ssim_list))


