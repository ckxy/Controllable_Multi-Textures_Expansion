import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import itertools


class N2Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        dir = os.path.join(opt.dataroot, 'train' if opt.isTrain else 'test')
        self.paths = make_dataset(dir)
        self.paths = sorted(self.paths)
        self.opt.n_pic = len(self.paths)
        assert self.opt.n_pic > 1
        print('#n_pic = %d' % len(self.paths))

        dir = os.path.join(opt.dataroot, 'trans')
        self.trans_paths = make_dataset(dir)
        self.trans_paths = sorted(self.trans_paths)

        self.fineSize = opt.fineSize
        self.transform = get_transform(opt)

        self.l = []
        for i in itertools.product([i for i in range(self.opt.n_pic)], repeat=2):
            self.l.append(i)
        print(self.l)

        self.t = []
        for i in itertools.permutations([i for i in range(self.opt.n_pic)], 2):
            self.t.append(i)
        print(self.t)

    def __getitem__(self, index):
        if self.l[index][0] == self.l[index][1]:
            B_path = self.paths[self.l[index][1]]
            B_img = Image.open(B_path).convert('RGB')
            if self.opt.isTrain and not self.opt.no_flip:
                if random.random() > 0.5:
                    B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)

            w, h = B_img.size
            rw = random.randint(0, w - self.fineSize)
            rh = random.randint(0, h - self.fineSize)
            B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))

            w, h = B_img.size
            rw = random.randint(0, int(w / 2))
            rh = random.randint(0, int(h / 2))

            A_img = B_img.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))
            A_star = A_img
        else:
            B_path = self.trans_paths[self.t.index(self.l[index])]
            A_path = self.paths[self.l[index][0]]
            B_img = Image.open(B_path).convert('RGB')
            A_img = Image.open(A_path).convert('RGB')

            if self.opt.isTrain and not self.opt.no_flip:
                if random.random() > 0.5:
                    B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)
                    A_img = A_img.transpose(Image.FLIP_LEFT_RIGHT)

            w, h = B_img.size
            rw = random.randint(0, w - self.fineSize)
            rh = random.randint(0, h - self.fineSize)
            B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))

            A_img = A_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
            w, h = A_img.size
            rw = random.randint(0, int(w / 2))
            rh = random.randint(0, int(h / 2))
            A_img = A_img.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))
            A_star = B_img.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))

        B_label_short = torch.LongTensor(1).zero_()
        B_label_short[0] = self.l[index][1]

        A_label_short = torch.LongTensor(1).zero_()
        A_label_short[0] = self.l[index][0]

        B_img = self.transform(B_img)
        A_img = self.transform(A_img)
        A_star = self.transform(A_star)

        return {'A': A_img, 'B': B_img, 'A_star': A_star,
                'A_label': A_label_short, 'B_label': B_label_short}

    def __len__(self):
        return len(self.l)

    def name(self):
        return 'N2Dataset'


class N2TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir = os.path.join(opt.dataroot, 'train')
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)

        self.fineSize = opt.fineSize
        self.transform = get_transform(opt)

        self.l = []
        for i in range(len(self.paths)):
            self.l.append((i, i))
        print(self.l)

    def __getitem__(self, index):
        B_path = self.paths[self.l[index][1]]
        A_path = B_path
        B_img = Image.open(B_path).convert('RGB')
        w, h = B_img.size
        rw = random.randint(0, w - self.fineSize)
        rh = random.randint(0, h - self.fineSize)
        B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
        A_img = B_img

        B_img = self.transform(B_img)
        A_img = self.transform(A_img)

        A_path = os.path.splitext(A_path)[0] + '->' + os.path.splitext(B_path[len(self.dir) + 1:])[0]

        return {'A': A_img, 'B': B_img, 'A_path': A_path}

    def __len__(self):
        return len(self.l)

    def name(self):
        return 'N2TestDataset'


class _2NDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        dir = os.path.join(opt.dataroot, 'train' if opt.isTrain else 'test')
        self.paths = make_dataset(dir)
        self.paths = sorted(self.paths)
        self.opt.n_pic = len(self.paths)
        assert self.opt.n_pic > 1
        print('#n_pic = %d' % len(self.paths))

        self.fineSize = opt.fineSize
        self.transform = get_transform(opt)

        self.l = []
        for i in range(self.opt.n_pic):
            self.l.append((i, i))
        for i in range(self.opt.n_pic):
            self.l.append((i, -1))
        print(self.l)

    def __getitem__(self, index):
        if self.l[index][0] == self.l[index][1]:
            B_path = self.paths[self.l[index][1]]
            B_img = Image.open(B_path).convert('RGB')
            if self.opt.isTrain and not self.opt.no_flip:
                if random.random() > 0.5:
                    B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)

            w, h = B_img.size
            rw = random.randint(0, w - self.fineSize)
            rh = random.randint(0, h - self.fineSize)
            B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))

            w, h = B_img.size
            rw = random.randint(0, int(w / 2))
            rh = random.randint(0, int(h / 2))

            A_img = B_img.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))

            # B_label_short = torch.LongTensor(1).zero_()
            # B_label_short[0] = self.l[index][1]

            B_label_short = torch.FloatTensor(self.opt.n_pic).zero_()
            B_label_short[self.l[index][1]] = 1

            A_label_short = B_label_short

            A_star = A_img
        else:
            dir = os.path.join(self.opt.dataroot, 'trans')
            dir = os.path.join(dir, str(self.l[index][0]))
            trans_paths = make_dataset(dir)
            trans_paths = sorted(trans_paths)

            # bi = random.randint(0, len(trans_paths) - 1)
            # B_path = trans_paths[bi]
            B_path = trans_paths[0]
            name = os.path.splitext(os.path.basename(B_path))[0]
            A_path = self.paths[self.l[index][0]]
            B_img = Image.open(B_path).convert('RGB')
            A_img = Image.open(A_path).convert('RGB')

            if self.opt.isTrain and not self.opt.no_flip:
                if random.random() > 0.5:
                    B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)
                    A_img = A_img.transpose(Image.FLIP_LEFT_RIGHT)

            w, h = B_img.size
            rw = random.randint(0, w - self.fineSize)
            rh = random.randint(0, h - self.fineSize)
            B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))

            A_img = A_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
            w, h = A_img.size
            rw = random.randint(0, int(w / 2))
            rh = random.randint(0, int(h / 2))
            A_img = A_img.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))
            A_star = B_img.crop((rw, rh, int(rw + w / 2), int(rh + h / 2)))

            # B_label_short = torch.LongTensor(1).zero_()
            # B_label_short[0] = int(name)
            #
            # A_label_short = torch.LongTensor(1).zero_()
            # A_label_short[0] = self.l[index][0]

            B_label_short = torch.FloatTensor(self.opt.n_pic).zero_()
            B_label_short[int(name)] = 1

            A_label_short = torch.FloatTensor(self.opt.n_pic).zero_()
            A_label_short[self.l[index][0]] = 1

        B_img = self.transform(B_img)
        A_img = self.transform(A_img)
        A_star = self.transform(A_star)

        return {'A': A_img, 'B': B_img, 'A_star': A_star,
                'A_label': A_label_short, 'B_label': B_label_short}

    def __len__(self):
        return len(self.l)

    def name(self):
        return '2NDataset'
