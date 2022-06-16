from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import random
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as trnasforms_f
import torch
import numpy as np
from glob import glob
import os

from baseline.modules.datasets import transform


#################################
# 레이블, 언레이블 훈련 이미지와 테스트 이미지 인덱스 설정
def get_harbor_idx(root, train = True, is_label=True, label_num = 15):
    if train:
        if is_label:
            classes = ['ship', 'container_truck', 'forklift', 'reach_stacker']
            image_path = glob(os.path.join(root, 'train', 'labeled_images', '*.jpg'))
            image_idx_list = list(map(lambda x: x.split('/')[-1].split('.')[0], image_path))
            train_idx = []
            valid_idx = []
            for c in classes:
                matched_idx = [i for i in image_idx_list if c in i]
                train_idx.extend(matched_idx[:label_num])
                valid_idx.extend(matched_idx[label_num:])
            return train_idx, valid_idx
        else:
            image_path = glob(os.path.join(root, 'train', 'unlabeled_images', '*.jpg'))
            train_idx = list(map(lambda x: x.split('/')[-1].split('.')[0], image_path))
            return train_idx
    else:
        image_path = glob(os.path.join(root, 'test', 'images', '*.jpg'))
        test_idx = list(map(lambda x: x.split('/')[-1].split('.')[0], image_path))
        return test_idx
        ###

##################
# 파이토치 포멧의 데이터셋 생성
class BuildDataset(Dataset):
    def __init__(self, root, idx_list, crop_size = (512, 512), scale_size = (0.5, 2.0),
                 augmentation = True, train = True, is_label = True):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.is_label = is_label

    def __getitem__(self, index):
        if self.train: # 훈련 데이터 로드
            if self.is_label:
                image_root = Image.open(self.root + f'/train/labeled_images/{self.idx_list[index]}.jpg')
                label_root = Image.open(self.root + f'/train/labels/{self.idx_list[index]}.png')
            else:
                image_root = Image.open(self.root + f'/train/unlabeled_images/{self.idx_list[index]}.jpg')
                label_root = None
            
            image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
            if label is not None:
                return image, label.sqeeze(0)
            else:
                return image

        else: # 모델 테스트
            file_name = f'{self.idx_list[index]}.jpg'
            image_root = Image.open(self.root + f'/test/images/{file_name}')
            image, label = transform(image_root, None, None, self.crop_size, self.scale_size, self.augmentation)
            return image, torch.tensor(image_root.size), file_name
    
    def __len__(self):
        return len(self.idx_list)
########

class BuildDataLoader:
    def __init__(self, num_labels, dataset_path, batch_size):
        self.data_path = dataset_path
        self.im_size = [513, 513]
        self.crop_size = [321, 321]
        self.num_segments = 5
        self.scale_size = (0.5, 1.5)
        self.batch_size = batch_size
        self.train_l_idx, self.valid_l_idx = get_harbor_idx(self.data_path, train = True, is_label=True, label_num=num_labels)
        self.train_u_idx = get_harbor_idx(self.data_path, train=True, is_label=False)
        self.test_idx = get_harbor_idx(self.data_path, train=False)

        if num_labels == 0: # using all data
            self.train_l_idx = self.train_u_idx
    
    def build(self, supervised=False, collate_fn = None):
        train_l_dataset = BuildDataset(self.data_path,self.train_l_idx, self.crop_size,
                                        scale_size=self.scale_size,
                                        augmentation=True, train=True, is_label=True)
        train_u_dataset = BuildDataset(self.data_path, self.train_u_idx, crop_size=self.crop_size,
                                        scale_size=self.scale_size, augmentation=False, train = True,
                                        is_label=False)
        valid_l_dataset = BuildDataset(self.data_path,self.valid_l_idx,
                                        crop_size=self.crop_size, scale_size=self.scale_size,
                                        augmentation=False, train=True, is_label=True)
        test_dataset = BuildDataset(self.data_path, self.test_idx, crop_size=self.im_size,
                                        scale_size=(1.0, 1.0), augmentation=False,
                                        train=False, is_label=True)

        if supervised: # no unlabelled dataset needed, double batch-size to match the same number of training samples
            self.batch_size = self.batch_size * 2

        num_samples = self.batch_size * 200 # for total 40k iterations with 200 epochs
        # num_samples = self.batch_size * e
        train

        return train_l_loader, train_u_loader, valid_l_loader, test_loader



###########
# Define data augmentation
###########
def transform(image, label = None, logits=None, crop_size = (512, 512),
              scale_size = (0.8, 1.0), augmentation = True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resize_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))