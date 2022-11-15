# 使用PaddleGAN中的NAFNet进行图像去模糊

## 1. 项目简介

### 1.1 项目来源
- 最近逛社区，偶有发现PaddleGan已经提供了模糊图片修复功能，并且开源了模型参数，经过一番调研后选取了尝试跑了一下NAFNet模型，先展示下效果：

处理前：
 <img src="https://github.com/Ulov888/PictureRestoration/blob/master/pictures/inputs/blurry-reds-1.jpg" width = "500" height = "300" alt="图片名称" align=center />

处理后：
 <img src="https://github.com/Ulov888/PictureRestoration/blob/master/pictures/outputs/blurry-reds-1_restoration.jpg" width = "500" height = "300" alt="图片名称" align=center />


是不是和我一样惊大了嘴巴:)，更重要的是，连模型参数也是开源的，下面给出完整的介绍以及运行方法




### 1.2 NAFNet介绍
- [NAFNet](https://arxiv.org/abs/2204.04676)是旷视研究院提出的用于图像复原的模型，在图像去模糊、去噪都取得了很好的性能，不仅**计算高效同时性能优于现有的SOTA方案**，效果如下图所示。在双目超分任务上，基于NAFNet的双目超分模型NAFSSR**获得NTIRE 2022的双目超分赛道冠军**

![](https://ai-studio-static-online.cdn.bcebos.com/04af7bb7c88a4b0ea780bd71b6a8e3d1a86f5b26d6a34113b9d204e2c66990d9)

## 2. 如何使用

- 首先我将NAFNet的deblur权重转换为Paddle的之后并挂载在项目的数据集中，一共有两个权重：
    - 在GoPro数据集上训练得到的`NAFNet-GoPro-width64.pdparams`， 主要用于运动模糊图像的去除
    - 在REDS数据集上训练得到的`NAFNet-REDS-width64.pdparams`，主要用于有压缩损失的模糊图像恢复，为方便下载，附上模型参数的网盘下载链接: https://pan.baidu.com/s/1lt4-DXjfHFrH9Yd7huknLA 提取码: phfk 
- 接下来则是基于PaddleGAN来调用该权重，运行本项目之前需要将PaddleGAN克隆至本地,并安装必要的依赖
    ```
    git clone https://github.com/PaddlePaddle/PaddleGAN
    
    cd PaddleGAN
    
    pip install -r requirements.txt

    ```
- 接下来只需要输入模糊图片所在文件夹和保存文件夹即可

    普通图片，在纯cpu环境一张图片处理速度约为1s左右
    ```
    python normal_pic_restoration.py --input_path "./pictures/inputs" --output_path "./pictures/outputs" --weight_path “NAFNet-REDS-width64.pdparams”
    ```
    4K图片，对于4k级别影像，直接预测会导致爆显存(Out of Memory, OOM)，所以要切块预测，我也封装好了，可以直接用
    ```
    python 4k_pic_restoration.py --input_path "./4kpictures/inputs" --output_path "./4kpictures/outputs" --weight_path “NAFNet-REDS-width64.pdparams”
    ```
- 现在就可以去看模糊修复效果啦，更多效果参见本项目下pictures/outputs和4kpictures/outputs

处理前：
 <img src="https://github.com/Ulov888/PictureRestoration/blob/master/pictures/inputs/3.png" width = "500" height = "300" alt="图片名称" align=center />

处理后：
 <img src="https://github.com/Ulov888/PictureRestoration/blob/master/pictures/outputs/3_restoration.png" width = "500" height = "300" alt="图片名称" align=center />
