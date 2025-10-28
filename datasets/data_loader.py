# 数据集加载器
import os
import numpy as np
from osgeo import gdal

import torch
from torch.utils.data import Dataset

def read_raster(infile):
    """读取栅格数据"""
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    fp = gdal.Open(infile)
    cols = fp.RasterXSize
    rows = fp.RasterYSize
    nb = fp.RasterCount
    if nb == 1:
        band = fp.GetRasterBand(1)
        data = band.ReadAsArray()
        data = data.reshape(1, rows, cols)
        band.GetScale()
        band.GetOffset()
        band.GetNoDataValue()
    else:
        data = np.zeros([nb, rows, cols])
        for i in range(0, nb):
            band = fp.GetRasterBand(i+1)
            data[i, :, :] = band.ReadAsArray()
            band.GetScale()
            band.GetOffset()
            band.GetNoDataValue()
    return rows, cols, nb, data

def read_tiff(filepath):
    """读取TIFF文件"""
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    bands = dataset.RasterCount
    height = dataset.RasterYSize
    width = dataset.RasterXSize
    image = np.zeros((bands, height, width), dtype=np.float32)
    
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        image[i, :, :] = band.ReadAsArray()
    
    dataset = None
    return image

def transform_image(image, flip_num, rotate_num0, rotate_num):
    """图像变换"""
    image_mask = np.ones(image.shape)
    negtive_mask = np.where(image < 0)
    inf_mask = np.where(image > 500.)

    image_mask[negtive_mask] = 0.0
    image_mask[inf_mask] = 0.0
    image[negtive_mask] = 0.0
    image[inf_mask] = 500.0
    image = image.astype(np.float32)

    if flip_num == 1:
        image = image[:, :, ::-1]

    C, H, W = image.shape
    if rotate_num0 == 1:
        if rotate_num == 2:
            image = image.transpose(0, 2, 1)[::-1, :]
        elif rotate_num == 1:
            image = image.transpose(0, 2, 1)[:, ::-1]
        else:
            image = image.reshape(C, H * W)[:, ::-1].reshape(C, H, W)

    image = torch.from_numpy(image.copy())
    image_mask = torch.from_numpy(image_mask)

    image.mul_(0.002)
    image = image * 2 - 1
    return image, image_mask

class PatchSet(Dataset):
    """标准数据集类"""
    def __init__(self, dates_num, generate_dir, type, image_size, patch_size):
        super(PatchSet, self).__init__()
        self.generate_dir = generate_dir
        self.dates_num = dates_num
        self.image_size = image_size
        self.patch_size = patch_size

        PATCH_STRIDE = self.patch_size // 2
        end_h = (self.image_size[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
        end_w = (self.image_size[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
        h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
        w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
        if (self.image_size[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
            h_index_list.append(self.image_size[0] - self.patch_size)
        if (self.image_size[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
            w_index_list.append(self.image_size[1] - self.patch_size)

        self.total_index = self.dates_num * len(h_index_list) * len(w_index_list)

    def __getitem__(self, item):
        images = []
        for i in range(4):
            tiff_file = os.path.join(self.generate_dir, f"{item}_{i}.tiff")
            im = read_tiff(tiff_file)
            images.append(im)

        patches = [None] * len(images)
        masks = [None] * len(images)

        flip_num = np.random.choice(2)
        rotate_num0 = np.random.choice(2)
        rotate_num = np.random.choice(3)

        for i in range(len(patches)):
            im = images[i]
            im, im_mask = transform_image(im, flip_num, rotate_num0, rotate_num)
            patches[i] = im
            masks[i] = im_mask

        gt_mask = masks[0] * masks[1] * masks[2] * masks[3]

        return patches[0], patches[1], patches[2], patches[3], gt_mask

    def __len__(self):
        return self.total_index

def get_pair_path(root_dir, target_dir_name, ref_dir_name):
    paths = [None, None, None, None]
    target_dir = root_dir + '/' + target_dir_name
    for filename in os.listdir(target_dir):
        if filename[:1] == 'M':
            paths[0] = os.path.join(target_dir, filename)
        else:
            paths[1] = os.path.join(target_dir, filename)

    target_dir = root_dir + '/' + ref_dir_name
    for filename in os.listdir(target_dir):
        if filename[:1] == 'M':
            paths[2] = os.path.join(target_dir, filename)
        else:
            paths[3] = os.path.join(target_dir, filename)

    return paths # C2, F2, C1, F1

def load_image_pair(root_dir, target_dir_name, ref_dir_name):
    paths = get_pair_path(root_dir, target_dir_name, ref_dir_name)
    images = []
    for p in paths:
        nl1, ns1, nb1, im = read_raster(p)
        images.append(im)

    return images
