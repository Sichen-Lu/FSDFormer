# Spilt the image into 256*256 patches and predict each patch.
# The predicted patches are then stitched back to the original image size.
import os
import argparse
import rasterio
import numpy as np
from sewar import rmse, ssim, sam
import cv2
import torch

from datasets.data_loader import load_image_pair, transform_image
from models.fsdformer import FSDFormer as Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def uiqi(im1, im2, block_size=64, return_map=False):
    if len(im1.shape) == 3:
        return np.array(
            [uiqi(im1[:, :, i], im2[:, :, i], block_size, return_map=return_map) for i in range(im1.shape[2])])
    delta_x = np.std(im1, ddof=1)
    delta_y = np.std(im2, ddof=1)
    delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / (im1.shape[0] * im1.shape[1] - 1)
    mu_x = np.mean(im1)
    mu_y = np.mean(im2)
    q1 = delta_xy / (delta_x * delta_y)
    q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2)
    q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2)
    q = q1 * q2 * q3
    return q

class ImageProcessor:
    def __init__(self, patch_size, image_size):
        self.patch_size = patch_size
        self.patch_stride = patch_size // 2
        self.image_size = image_size
        self.h_index_list, self.w_index_list = self._generate_indices()

    def _generate_indices(self):
        end_h = (self.image_size[0] - self.patch_stride) // self.patch_stride * self.patch_stride
        end_w = (self.image_size[1] - self.patch_stride) // self.patch_stride * self.patch_stride
        h_index_list = [i for i in range(0, end_h, self.patch_stride)]
        w_index_list = [i for i in range(0, end_w, self.patch_stride)]
        if (self.image_size[0] - self.patch_stride) % self.patch_stride != 0:
            h_index_list.append(self.image_size[0] - self.patch_size)
        if (self.image_size[1] - self.patch_stride) % self.patch_stride != 0:
            w_index_list.append(self.image_size[1] - self.patch_size)
        return h_index_list, w_index_list

    def cut_and_save_images(self, images, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for idx, (h_start, w_start) in enumerate([(h, w) for h in self.h_index_list for w in self.w_index_list]): 
            patches = []
            for img in images:
                
                patch = img[:4, h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
                patches.append(patch)
            self._save_patch(patches, save_dir, idx, img_num=len(images))
        return 

    def _save_patch(self, patch, save_dir, patch_id, img_num):
        metadata = {
            'driver': 'GTiff',
            'width': patch[0].shape[2],
            'height': patch[0].shape[1],
            'count': img_num*4,
            'dtype': np.int16,
            'compress': 'lzw',
        }
        file_path = os.path.join(save_dir, f'patch_{patch_id}.tif')
        with rasterio.open(file_path, 'w', **metadata) as dst:
            for i in range(len(patch)):
                # dst.write(patch[i], 1)
                for j in range(patch[0].shape[0]):
                    dst.write(patch[i][j].astype(np.int16), i*4 + j + 1)              

    def load_patches(self, load_dir):
        patch_files = ['patch_' + str(i) + '.tif' for i in range(len(self.h_index_list) * len(self.w_index_list))]
        patches = [rasterio.open(os.path.join(load_dir, f)).read() for f in patch_files]
        return patches

    def reassemble_image(self, patches):
        output_image = np.zeros((patches[0].shape[0], self.image_size[0], self.image_size[1]))
        patch_idx = 0
        for i in range(len(self.h_index_list)):
            for j in range(len(self.w_index_list)):
                h_start = self.h_index_list[i]
                w_start = self.w_index_list[j]
                h_end = h_start + self.patch_size
                w_end = w_start + self.patch_size

                cur_h_start = cur_w_start = 0
                cur_h_end = cur_w_end = self.patch_size

                if i != 0:
                    h_start += self.patch_size // 4
                    cur_h_start = self.patch_size // 4
                if i != len(self.h_index_list) - 1:
                    h_end -= self.patch_size // 4
                    cur_h_end -= self.patch_size // 4
                if j != 0:
                    w_start += self.patch_size // 4
                    cur_w_start = self.patch_size // 4
                if j != len(self.w_index_list) - 1:
                    w_end -= self.patch_size // 4
                    cur_w_end -= self.patch_size // 4

                output_image[:, h_start:h_end, w_start:w_end] = \
                    patches[patch_idx][:, cur_h_start:cur_h_end, cur_w_start:cur_w_end]
                patch_idx += 1
        return output_image


def test(opt, model, test_dates, image_processor):
    cur_result = {}
    model.eval()

    total_image = 0
    for cur_date in test_dates:
        if cur_date == CUR_DATE:
            for ref_date in test_dates:
                if ref_date != cur_date:
                    
                    patches = image_processor.load_patches(f'datasets/patches_DX/{cur_date}_{ref_date}')
                    output_patches = []
                    F2_out = []

                    for patch_group in patches:
                        C2 = patch_group[0:4, :, :]
                        F2 = patch_group[4:8, :, :]
                        C1 = patch_group[8:12, :, :]
                        F1 = patch_group[12:16, :, :]

                        flip_num = 0
                        rotate_num0 = 0
                        rotate_num = 0
                        C2, _ = transform_image(C2, flip_num, rotate_num0, rotate_num)
                        C1, _ = transform_image(C1, flip_num, rotate_num0, rotate_num)
                        F1, _ = transform_image(F1, flip_num, rotate_num0, rotate_num)

                        C2 = C2.unsqueeze(0).cuda()
                        C1 = C1.unsqueeze(0).cuda()
                        F1 = F1.unsqueeze(0).cuda()

                        result = model(F1, C2, C1)

                        output_patches.append(result.squeeze().cpu().detach().numpy())
                        F2_out.append(F2)

                    output_image = image_processor.reassemble_image(output_patches)
                    real_im = image_processor.reassemble_image(F2_out) * 0.002
                    real_output = (output_image + 1) * 0.5

                    for real_predict in [real_output]:
                        cur_result['rmse'] = []
                        cur_result['ssim'] = []
                        cur_result['cc'] = []
                        cur_result['uiqi'] = []
                        cur_result['ergas'] = 0

                        for i in range(4):
                            cur_result['rmse'].append(rmse(real_im[i], real_predict[i]))
                            cur_result['ssim'].append(ssim(real_im[i], real_predict[i], MAX=1.0)[0])
                            cur_result['uiqi'].append(uiqi(real_im[i], real_predict[i]))
                            cur_cc = np.sum(
                                (real_im[i] - np.mean(real_im[i])) * (real_predict[i] - np.mean(real_predict[i]))) / \
                                     np.sqrt((np.sum(np.square(real_im[i] - np.mean(real_im[i])))) * np.sum(
                                         np.square(real_predict[i] - np.mean(real_predict[i]))) + 1e-100)
                            cur_result['cc'].append(cur_cc)

                            cur_result['ergas'] += rmse(real_im[i], real_predict[i]) ** 2 / (
                                        np.mean(real_im[i]) ** 2 + 1e-100)

                        cur_result['ergas'] = np.sqrt(cur_result['ergas'] / 4.) * 4

                        cur_im = real_im * 500.
                        cur_predict = real_predict * 500.

                        cur_result['sam'] = sam(cur_im.transpose(1, 2, 0), cur_predict.transpose(1, 2, 0)) * 180 / np.pi
                        print('[%s/%s] RMSE: %.4f SSIM: %.4f UIQI: %.4f CC: %.4f ERGAS: %.4f SAM: %.4f' % (
                            cur_date, ref_date, np.mean(np.array(cur_result['rmse'])),
                            np.mean(np.array(cur_result['ssim'])), np.mean(np.array(cur_result['uiqi'])),
                            np.mean(np.array(cur_result['cc'])), cur_result['ergas'], cur_result['sam']
                        ))
                        total_image += cur_predict
                        if ref_date != CUR_DATE:
                            final_im = cur_predict.astype(np.int16)
                            metadata = {
                                'driver': 'GTiff',
                                'width': final_im.shape[2],
                                'height': final_im.shape[1],
                                'count': final_im.shape[0],
                                'dtype': np.int16
                            }
                            save_dir = os.path.join('./show/DX/FSDFormer', CUR_DATE)
                            print(save_dir)
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            im_name = os.path.join(save_dir, 'result_' + cur_date + '_' + ref_date + '.tif')
                            im_result = os.path.join(save_dir, 'result_' + cur_date + '_' + ref_date + '.png')

                            png = final_im.transpose(1, 2, 0)
                            png = png[:, :, 0:3] / png.max() * 255


                            cv2.imwrite(im_result, png)
                            assert final_im.ndim == 2 or final_im.ndim == 3
                            with rasterio.open(im_name, mode='w', **metadata) as dst:
                                if final_im.ndim == 3:
                                    for i in range(final_im.shape[0]):
                                        dst.write(final_im[i], i + 1)
                                else:
                                    dst.write(final_im, 1)

CUR_DATE = '20170507'

def main():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--image_size', default=[1640,1640], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=256, type=int, help='training images crop size')
    parser.add_argument('--root_dir', default='Daxing_test', help='Datasets root directory')

    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size
    image_processor = ImageProcessor(PATCH_SIZE, IMAGE_SIZE)

    test_dates = ['20161214', '20170507']

    # Spilt the images into patches and save them.
    for cur_date in test_dates:
        for ref_date in test_dates:
            if ref_date != cur_date and cur_date == CUR_DATE:
                images = load_image_pair(opt.root_dir, cur_date, ref_date)
                save_dir = f'datasets/patches_DX/{cur_date}_{ref_date}'
                image_processor.cut_and_save_images(images, save_dir)

    model_G = Model()
    model_G = torch.nn.DataParallel(model_G)
    G_dict = model_G.state_dict()
    model_CKPT = torch.load('./experiments/FSDFormer_Transformer_Daxing.pth')

    pretained_dict = {k: v for k, v in model_CKPT.items() if k in G_dict}
    G_dict.update(pretained_dict)
    model_G.load_state_dict(G_dict)
    model_G.cuda()

    test(opt, model_G, test_dates, image_processor)


if __name__ == '__main__':
    main()