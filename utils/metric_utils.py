from diff_rendering.gaussian_splatting_2d.utils.image_utils import psnr
from utils.quaternion_utils import quaternion_standardize
import torch
import torch.nn.functional as F

def mean_psnr(images,gt_images):
    psnr_mean = 0.0
    for image,gt_image in zip(images,gt_images):
        # Ensure both tensors are on the same device (move gt_image to image's device)
        if image.device != gt_image.device:
            gt_image = gt_image.to(image.device)
        psnr_mean += psnr(image, gt_image).mean().double()
    psnr_mean /= len(images)
    return psnr_mean
