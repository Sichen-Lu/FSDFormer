import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--image_size', default=[1640, 1640], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=256, type=int, help='training images crop size')
    parser.add_argument('--root_dir', default='Daxing', help='Datasets root directory')

    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size

    train_dates = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11','12',
                   '18','19','20','21','22','23','24','25','26','27','28','29'] # Dates of training, sort by ascending order
    # train_dates = ['13','14','15','16','17'] # Dates of validation

    # split the whole image into several patches
    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]

    if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
    if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

    total_index = 0
    # path where the training images saved in
    output_dir = 'Daxing_Train' # Training image directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save all the train images into one numpy array
    total_original_images = np.zeros((len(train_dates), 2, 6, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    for k in tqdm(range(len(train_dates))):
        cur_date = train_dates[k]
        target_dir = os.path.join(opt.root_dir, cur_date)
        for filename in os.listdir(target_dir):
            if filename[:1] != 'M':
                path = os.path.join(target_dir, filename)
                data_temp = gdal.Open(path)
                total_original_images[k, 1] = data_temp.ReadAsArray()
            else:
                path = os.path.join(target_dir, filename)
                data_temp = gdal.Open(path)
                total_original_images[k, 0] = data_temp.ReadAsArray()

    for k in tqdm(range(len(train_dates))):
        for i in range(len(h_index_list)):
            for j in range(len(w_index_list)):
                h_start = h_index_list[i]
                w_start = w_index_list[j]

                ref_index = k
                while ref_index == k:
                    ref_index = np.random.choice(len(train_dates))

                images = []
                images.append(total_original_images[k, 0])
                images.append(total_original_images[k, 1])
                images.append(total_original_images[ref_index, 0])
                images.append(total_original_images[ref_index, 1])

                for idx, im in enumerate(images):
                    patch = im[:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                    patch_filename = f"{total_index}_{idx}.tiff"
                    patch_filepath = os.path.join(output_dir, patch_filename)
                    save_tiff(patch, patch_filepath)

                total_index += 1

    assert total_index == len(train_dates) * len(h_index_list) * len(w_index_list)


def save_tiff(image, filepath):
    driver = gdal.GetDriverByName("GTiff")

    if len(image.shape) == 3:
        bands, height, width = image.shape
    else:
        bands = 1
        height, width = image.shape
        image = np.expand_dims(image, axis=0)

    dataset = driver.Create(filepath, width, height, bands, gdal.GDT_Int16)
    dataset.SetMetadataItem('COMPRESSION', 'LZW')  

    for i in range(bands):
        band_data = image[i].astype(np.int16)
        dataset.GetRasterBand(i + 1).WriteArray(band_data)

    dataset.FlushCache()
    dataset = None


if __name__ == '__main__':
    main()
