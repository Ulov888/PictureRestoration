import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

import paddle


from normal_pic_restoration import NAFNetDeblurer

class CropPredictor(NAFNetDeblurer):
    def __init__(self,
                 output_path='output_dir',
                 weight_path=None):
        super(CropPredictor, self).__init__(output_path, weight_path)

    def crop_predict(self, img_lq):
        sf = self.sf
        tile = self.tile
        overlap = self.overlap
        b, c, h, w = img_lq.shape
        tile_overlap = overlap
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = paddle.zeros([b, c, h*sf, w*sf], dtype=img_lq.dtype)
        W = paddle.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                h_idx = int(h_idx)
                w_idx = int(w_idx)
                in_patch = img_lq[:, :,h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = self.generator(in_patch)
                out_patch_mask = paddle.ones_like(out_patch)

                E[:, :, h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch
                W[:, :, h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch_mask

        output = E.divide(W)
        return output

    def run_patches(self, images_path=None, tile=1024, overlap=128):
        os.makedirs(self.output_path, exist_ok=True)
        task_path = os.path.join(self.output_path, self.task)
        os.makedirs(task_path, exist_ok=True)
        image_files = self.get_images(images_path)
        self.tile = tile
        self.overlap = overlap
        self.sf = 1

        for image_file in tqdm(image_files):
            img_L = self.imread_uint(image_file, 3)

            image_name = os.path.basename(image_file)
            img = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(task_path, image_name), img)

            tmps = image_name.split('.')
            assert len(
                tmps) == 2, f'Invalid image name: {image_name}, too much "."'
            restoration_save_path = os.path.join(
                task_path, f'{tmps[0]}_restoration.{tmps[1]}')

            img_L = self.uint2single(img_L)

            # HWC to CHW, numpy to tensor
            img_L = self.single2tensor3(img_L)
            img_L = img_L.unsqueeze(0)
            with paddle.no_grad():
                output = self.crop_predict(img_L)

            restored = paddle.clip(output, 0, 1)

            restored = restored.numpy()
            restored = restored.transpose(0, 2, 3, 1)
            restored = restored[0]
            restored = restored * 255
            restored = restored.astype(np.uint8)

            cv2.imwrite(restoration_save_path,
                        cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

        print('Done, output path is:', task_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="picture_restoration")
    parser.add_argument("--input_path", type=str, default="./4kpictures/inputs", help="定义输入路径")
    parser.add_argument("--output_path", type=str, default="./4kpictures/outputs", help="定义输出路径")
    parser.add_argument("--weight_path", type=str, default="./models/NAFNet-REDS-width64.pdparams", help="定义权重所在路径")
    args = parser.parse_args()



    # 定义去模糊类
    croppredictor = NAFNetDeblurer(args.output_path, args.weight_path)
    # 执行预测
    croppredictor.run(images_path=args.input_path)