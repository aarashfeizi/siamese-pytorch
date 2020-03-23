import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

RESHAPE_SIZE = (105, 105)


def loadCUBToMem(dataPath, subPath, isTrain=True):
    isTrain = int(isTrain)  # True: 1, False: 0
    if subPath is not None:
        imagePath = os.path.join(dataPath, subPath)
    else:
        imagePath = dataPath
    print("begin loading dataset to memory")
    datas = {}

    with open(os.path.join(dataPath, 'train_test_split.txt'), 'r') as f:  # TODO
        train_test_split = np.array(list(map(lambda x: x.split(), f.read().split('\n')[:-1])), dtype=int)

    with open(os.path.join(dataPath, 'images.txt'), 'r') as f:
        idx2path = dict(list(map(lambda x: x.split(), f.read().split('\n')[:-1])))

    data_idx = train_test_split[train_test_split[:, 1] == isTrain]
    num_instances = data_idx.shape[0]

    num_classes = 1
    for idx in data_idx:
        path = idx2path[str(idx[0])]
        filePath = os.path.join(imagePath, path)
        data_class = int(path[0:3]) - 1
        if data_class not in datas.keys():
            datas[data_class] = []

        datas[data_class].append(filePath)

    if num_classes < data_class + 1:
        num_classes = data_class + 1

    print("finish loading dataset to memory")
    return datas, num_classes, num_instances


class CUBTrain(Dataset):

    def __init__(self, args, transform=None):
        super(CUBTrain, self).__init__()
        np.random.seed(args.seed)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes, self.length = loadCUBToMem(args.dataset_path, args.subdir_path, isTrain=True)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = Image.open(random.choice(self.datas[idx1]))
            image2 = Image.open(random.choice(self.datas[idx1]))
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = Image.open(random.choice(self.datas[idx1]))
            image2 = Image.open(random.choice(self.datas[idx2]))

        image1 = image1.resize(RESHAPE_SIZE).convert('RGB')
        image2 = image2.resize(RESHAPE_SIZE).convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class CUBTest(Dataset):

    def __init__(self, args, transform=None):
        np.random.seed(args.seed)
        super(CUBTest, self).__init__()
        self.transform = transform
        self.times = args.times
        self.way = args.way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes, _ = loadCUBToMem(args.dataset_path, args.subdir_path, isTrain=False)

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = Image.open(random.choice(self.datas[self.c1])).resize(RESHAPE_SIZE).convert('RGB')
            img2 = Image.open(random.choice(self.datas[self.c1])).resize(RESHAPE_SIZE).convert('RGB')
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = Image.open(random.choice(self.datas[c2])).resize(RESHAPE_SIZE).convert('RGB')

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


class OmniglotTrain(Dataset):

    def __init__(self, args, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(args.seed)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(args.train_path)
        # import pdb
        # pdb.set_trace()

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        agrees = [0, 90, 180, 270]
        idx = 0
        for agree in agrees:
            for alphaPath in os.listdir(dataPath):
                for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                    datas[idx] = []
                    for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                        filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                        datas[idx].append(Image.open(filePath).rotate(agree).convert('L'))
                    idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self):
        return 21000000

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class OmniglotTest(Dataset):

    def __init__(self, args, transform=None):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = args.times
        self.way = args.way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(args.test_path)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        return img1, img2


# test
if __name__ == '__main__':
    omniglotTrain = OmniglotTrain('./images_background', 30000 * 8)
    print(omniglotTrain)
