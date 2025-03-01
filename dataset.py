"""
vimeo dataset and general dataset for video compression
"""

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from time import perf_counter
from torch.nn.functional import pad
import numpy as np
import torch
import torchvision.transforms as T
import os
import math
import cv2


class Vimeo(Dataset):
    def __init__(self, data_path:Path, num_GOP, if_train):
        super(Vimeo, self).__init__()

        input_data = []
        self.if_train = if_train
        dir_name = os.listdir(data_path)

        for line in dir_name:
            subdir_name = os.listdir(data_path / line)
            input_data += [str(data_path / line / name) for name in subdir_name]

        self.image_input_list = input_data
        self.num_GOP = num_GOP

    def __len__(self):
        return len(self.image_input_list)
    
    def get_transform(self, train_flag):
        if train_flag:
            transform = T.Compose([
                T.RandomCrop(256, pad_if_needed=True),
                # T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                T.ToTensor()
            ])
        else:
            transform = T.Compose([T.ToTensor()])
        return transform

    def __getitem__(self, index):
        flag = True
        transform = self.get_transform(self.if_train)
        for index_inter in range(1, self.num_GOP + 1):
            image_path = self.image_input_list[index] + f"/im{index_inter}.png"
            image = Image.open(image_path).convert("RGB")
            transformed_image = transform(image).unsqueeze(1)
            if flag:
                flag = False
                out_image = transformed_image
            else:
                out_image = torch.concat((out_image, transformed_image), dim=1)
        
        return out_image
    

class Video():
    def __init__(self, video_path, downsample_scale, length_GOP=4, mode="RGB"):
        assert os.path.exists(video_path), f"Nothing found in {video_path}"
        assert mode in ("RGB", "RAW"), f"Unrecognized read mode for {mode}"
        self.capture = cv2.VideoCapture(video_path)
        self.mode = mode
        if self.mode == "RAW":
            self.capture.set(cv2.CAP_PROP_FORMAT, -1)
        self.width = int(self.capture.get(3))
        self.height = int(self.capture.get(4))
        self.codec = self.capture.get(cv2.CAP_PROP_FOURCC)
        self.fps = self.capture.get(5)
        self.num_frame = int(self.capture.get(7))
        self.length_GOP = length_GOP
        self.current_pointer = 0
        self.padded_width = 0
        self.padded_height = 0

        if not self.width % downsample_scale == 0:
            num = math.ceil(self.width / downsample_scale)
            self.padded_width = int(num * downsample_scale - self.width)

        if not self.height % downsample_scale == 0:
            num = math.ceil(self.height / downsample_scale)
            self.padded_height = int(num * downsample_scale - self.height)

        print(f"----------ready to load {self.num_frame} frames from {video_path}----------")

    def get_frame(self, if_tensor):
        assert self.capture.isOpened(), f"current video stream is closed"
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_pointer)
        frame_slice = []
        t1 = perf_counter()
        for n in range(self.length_GOP):
            ret, frame = self.capture.read()
            if not ret:
                break
            else:
                if self.mode == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slice.append(frame)
                self.current_pointer += 1
        t2 = perf_counter()
        print(
            f"read frames [{self.current_pointer - self.length_GOP + 1} -- {self.current_pointer}] / [{self.num_frame}] and it costs {round((t2 - t1) * 1000, ndigits=3)} ms")

        if if_tensor:
            tensor_frame = torch.as_tensor(np.array(frame_slice),
                                           dtype=torch.float32)  # Creating a tensor from a list of numpy.ndarrays is extremely slow.
            # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
            tensor_frame = torch.permute(tensor_frame, (0, 3, 1, 2))
            tensor_frame /= 255.

            tensor_frame = pad(tensor_frame, [self.padded_width, 0, self.padded_height, 0], mode="constant",
                               value=0.)  # [left, right, top, bottom]
            # tensor_frame = tensor_frame[:, :, 0:256, 0:256]
            return tensor_frame
        else:
            return frame_slice

    def clear(self):
        self.capture.release()

    def get_attribution(self):
        return dict(
            width=self.width,
            height=self.height,
            fps=self.fps,
            num_frame=self.num_frame,
            length_GOP=self.length_GOP
        )
        
        
        
        
'------openimages的数据处理  LSCI中的--------------------'
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance
import numpy as np
import torchvision.transforms as transforms
class Openimage(Dataset):
    def __init__(self,conf_path, new_size):
        # self.openimage_path = "/usr/Datasets/Openimage"#训练
        # self.openimage_path = "/usr/Datasets/Openimage_test"#test
        self.image_path = conf_path
        # self.openimage_path = "/home/pawup/Desktop/picture_test"#出论文图的12张
        self.dir = os.listdir(self.image_path)
        self.new_size = new_size

    def __len__(self):#30609
        length = len(self.dir)
        # length = 200000  #控制读取的图片长度
        return length

    def __getitem__(self, index):
        img_sz = self.new_size
        '''这一行是处理openimages数据集的'''
        # image = Image.open(self.image_path+ "/" + self.dir[index]) 
        '''这2行是处理cityscapes数据集的'''
        image_path = str(self.image_path / self.dir[index])
        image = Image.open(image_path)
        
        RandomCrop = transforms.RandomCrop((img_sz, img_sz)) #裁剪

        iw, ih = image.size
        if iw < img_sz or ih < img_sz:
            image = image.resize((img_sz, img_sz))
        else:
            image = RandomCrop(image)
        image = np.array(image, dtype=np.uint8)


        if len(image.shape)<3:
            image = image.reshape(img_sz, img_sz, 1)
            image = np.repeat(image, 3, axis=2)
        if image.shape[2]>3: #[0]h垂直尺寸；[1]w水平尺寸；[2]c通道尺寸。 [:2]取h和w；[:3]取高、宽、通道
            image = image[:, :, :3]

        image_t = np.transpose(image,(2,0,1))
        image_t = image_t/255 #像素[0, 255]变为[0, 1]

        return image_t


class Cityscapes_data(Dataset):
    def __init__(self,conf_path, new_width):
        # self.openimage_path = "/usr/Datasets/Openimage"#训练
        # self.openimage_path = "/usr/Datasets/Openimage_test"#test
        self.image_path = conf_path
        # self.openimage_path = "/home/pawup/Desktop/picture_test"#出论文图的12张
        self.dir = os.listdir(self.image_path)
        self.new_size = new_width

    def __len__(self):#30609
        length = len(self.dir)
        # length = 200000  #控制读取的图片长度
        return length

    def __getitem__(self, index):
        new_width = self.new_size
        '''这一行是处理openimages数据集的'''
        # image = Image.open(self.image_path+ "/" + self.dir[index]) 
        '''这2行是处理cityscapes数据集的'''
        image_path = str(self.image_path / self.dir[index])
        image = Image.open(image_path)
        
        # 按照原图比例调整大小
        width_percent = (new_width / float(image.size[0]))
        new_height = int((float(image.size[1]) * float(width_percent)))
        image = image.resize((new_width, new_height), Image.BILINEAR) # 使用双线性插值进行resize

        image = np.array(image, dtype=np.uint8)

        if len(image.shape)<3:
            image = image.reshape(new_height, new_width, 1)
            image = np.repeat(image, 3, axis=2)
        if image.shape[2]>3: #[0]h垂直尺寸；[1]w水平尺寸；[2]c通道尺寸。 [:2]取h和w；[:3]取高、宽、通道
            image = image[:, :, :3]

        image_t = np.transpose(image,(2,0,1))
        image_t = image_t/255 #像素[0, 255]变为[0, 1]

        return image_t


'''要实现图像的缩放并保留图像中的整体特征，同时将图像填充为固定的 new_width，
你可以先将图像按比例缩放到 new_width 的宽度，
然后通过填充使其达到 new_width x new_width 的尺寸。填充时通常使用黑色背景（或其他颜色）'''
class  PALM_data(Dataset):
    def __init__(self,conf_path, new_width):
        # self.openimage_path = "/usr/Datasets/Openimage"#训练
        # self.openimage_path = "/usr/Datasets/Openimage_test"#test
        self.image_path = conf_path
        # self.openimage_path = "/home/pawup/Desktop/picture_test"#出论文图的12张
        self.dir = os.listdir(self.image_path)
        self.new_size = new_width

    def __len__(self):#30609
        length = len(self.dir)
        # length = 200000  #控制读取的图片长度
        return length

    def __getitem__(self, index):
        new_width = self.new_size
        '''这一行是处理openimages数据集的'''
        # image = Image.open(self.image_path+ "/" + self.dir[index]) 
        '''这一行是处理PALM数据集的'''
        image_ini = Image.open(self.image_path / self.dir[index])
        # print("image_ini_size:", image_ini.size)  #(2124, 2056)
        # '''这2行是处理cityscapes数据集的'''
        # image_path = str(self.image_path / self.dir[index])
        # image = Image.open(image_path)
        
        # 按照原图比例调整大小，使宽度为 new_width，同时保持原始宽高比
        width_percent = (new_width / float(image_ini.size[0]))
        new_height = int((float(image_ini.size[1]) * float(width_percent)))
        image_resized = image_ini.resize((new_width, new_height), Image.BILINEAR) # 使用双线性插值进行resize
        # print("image_resized_size:", image_resized.size)  
        
        # 创建一个新的空白图像，并填充为黑色 (0, 0, 0)
        image_padded = Image.new("RGB", (new_width, new_width), (0, 0, 0))
        # 将缩放后的图像放置到新的图像中央
        image_padded.paste(image_resized, ((new_width - image_resized.width) // 2,
                                           (new_width - image_resized.height) // 2))
        # 将图像转换为numpy数组        
        image = np.array(image_padded, dtype = np.uint8)
        # print("image_size:", image.size)  #(1444, 1444)
        
        # 如果是灰度图像，转换为RGB
        if len(image.shape)<3:
            image = image.reshape(new_height, new_width, 1)
            image = np.repeat(image, 3, axis=2)
        # 如果通道数大于3，取前三个通道
        if image.shape[2]>3: #[0]h垂直尺寸；[1]w水平尺寸；[2]c通道尺寸。 [:2]取h和w；[:3]取高、宽、通道
            image = image[:, :, :3]

        # 转换为PyTorch的张量格式，并进行归一化处理
        image_t = np.transpose(image,(2,0,1)) # HWC -> CHW
        image_t = image_t/255 #像素[0, 255]变为[0, 1]

        return image_t
