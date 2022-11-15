import cv2
from glob import glob
from natsort import natsorted
import numpy as np
import os
from tqdm import tqdm
import argparse

import paddle

from PaddleGAN.ppgan.models.generators import NAFNetLocal
from PaddleGAN.ppgan.apps.base_predictor import BasePredictor

# 模型参数定义
model_cfgs = {
    'Deblur': {
        'img_channel': 3,
        'width': 64,
        'enc_blk_nums': [1, 1, 1, 28],
        'middle_blk_num': 1,
        'dec_blk_nums': [1, 1, 1, 1]
    }
}

# 定义去模糊的预测类
class NAFNetDeblurer(BasePredictor):

    def __init__(self,
                 output_path='output_dir',
                 weight_path=None):
        self.output_path = output_path
        task = 'Deblur'
        self.task = task

        checkpoint = paddle.load(weight_path)

        self.generator = NAFNetLocal(
            img_channel=model_cfgs[task]['img_channel'],
            width=model_cfgs[task]['width'],
            enc_blk_nums=model_cfgs[task]['enc_blk_nums'],
            middle_blk_num=model_cfgs[task]['middle_blk_num'],
            dec_blk_nums=model_cfgs[task]['dec_blk_nums'])

        self.generator.set_state_dict(checkpoint)
        self.generator.eval()

    def get_images(self, images_path):
        if os.path.isdir(images_path):
            return natsorted(
                glob(os.path.join(images_path, '*.jpeg')) +
                glob(os.path.join(images_path, '*.jpg')) +
                glob(os.path.join(images_path, '*.JPG')) +
                glob(os.path.join(images_path, '*.png')) +
                glob(os.path.join(images_path, '*.PNG')))
        else:
            return [images_path]

    def imread_uint(self, path, n_channels=3):
        #  input: path
        # output: HxWx3(RGB or GGG), or HxWx1 (G)
        if n_channels == 1:
            img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
            img = np.expand_dims(img, axis=2)  # HxWx1
        elif n_channels == 3:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        return img

    def uint2single(self, img):

        return np.float32(img / 255.)

    # convert single (HxWxC) to 3-dimensional paddle tensor
    def single2tensor3(self, img):
        return paddle.Tensor(np.ascontiguousarray(
            img, dtype=np.float32)).transpose([2, 0, 1])

    def run(self, images_path=None):
        os.makedirs(self.output_path, exist_ok=True)
        task_path = os.path.join(self.output_path, self.task)
        os.makedirs(task_path, exist_ok=True)
        image_files = self.get_images(images_path)
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
                output = self.generator(img_L)

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
    parser.add_argument("--input_path", type=str, default="./pictures/inputs", help="定义输入路径")
    parser.add_argument("--output_path", type=str, default="./pictures/outputs", help="定义输出路径")
    parser.add_argument("--weight_path", type=str, default="./models/NAFNet-REDS-width64.pdparams", help="定义权重所在路径")
    args = parser.parse_args()



    # 定义去模糊类
    deblur_predictor = NAFNetDeblurer(args.output_path, args.weight_path)
    # 执行预测
    deblur_predictor.run(images_path=args.input_path)