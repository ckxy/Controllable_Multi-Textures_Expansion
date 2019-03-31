import os
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import itertools


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.train_dir = os.path.join(opt.dataroot, 'train')
        self.dir_A = os.path.join(opt.dataroot, opt.which_content)
        self.dir_B = os.path.join(self.train_dir, opt.which_style)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_img = Image.open(self.dir_A).convert('RGB')
        B_img = Image.open(self.dir_B).convert('RGB')
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img, 'A_path': ''}

    def __len__(self):
        return 1

    def name(self):
        return 'SingleDataset'


class SingleN2Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir = os.path.join(opt.dataroot, 'train')
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)

        self.transform = get_transform(opt)

        self.l = []
        for i in itertools.product([i for i in range(len(self.paths))], repeat=2):
            self.l.append(i)
        print(self.l)

    def __getitem__(self, index):
        if self.l[index][0] == self.l[index][1]:
            B_path = self.paths[self.l[index][1]]
            A_path = B_path
            B_img = Image.open(B_path).convert('RGB')
            A_img = B_img
        else:
            B_path = self.paths[self.l[index][1]]
            A_path = self.paths[self.l[index][0]]
            B_img = Image.open(B_path).convert('RGB')
            A_img = Image.open(A_path).convert('RGB')

        B_img = self.transform(B_img)
        A_img = self.transform(A_img)

        A_path = os.path.splitext(A_path)[0] + '->' + os.path.splitext(B_path[len(self.dir) + 1:])[0]

        # s = 64
        # B_img = B_img[:, :s, :s]

        return {'A': A_img, 'B': B_img, 'A_path': A_path}

    def __len__(self):
        return len(self.l)

    def name(self):
        return 'SingleN2Dataset'


class SingleNDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir = os.path.join(opt.dataroot, 'train')
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)

        self.test_dir = os.path.join(opt.dataroot, 'test')
        self.test_paths = make_dataset(self.test_dir)
        self.test_paths = sorted(self.test_paths)

        self.transform = get_transform(opt)

        self.l = []
        for i in itertools.product([i for i in range(len(self.test_paths))], [i for i in range(len(self.paths))]):
            self.l.append(i)
        print(self.l)

    def __getitem__(self, index):
        B_path = self.paths[self.l[index][1]]
        A_path = self.test_paths[self.l[index][0]]
        B_img = Image.open(B_path).convert('RGB')
        A_img = Image.open(A_path).convert('RGB')

        A_path = os.path.splitext(A_path)[0] + '->' + os.path.splitext(B_path[len(self.dir) + 1:])[0]

        B_img = self.transform(B_img)
        A_img = self.transform(A_img)

        # B_img = B_img[:, :200, -200:]

        return {'A': A_img, 'B': B_img, 'A_path': A_path}

    def __len__(self):
        return len(self.l)

    def name(self):
        return 'SingleNDataset'
