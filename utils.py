import argparse

import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
import numpy as np


class TransformLoader:
    def __init__(self, image_size, rotate=0,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        self.rotate = rotate

    def parse_transform(self, transform_type):
        # if transform_type == 'ImageJitter':
        #     method = add_transforms.ImageJitter(self.jitter_param)
        #     return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Scale':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        elif transform_type == 'RandomRotation':
            return method(self.rotate)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        elif not aug and self.rotate == 0:
            transform_list = ['Scale', 'CenterCrop', 'ToTensor', 'Normalize']
        elif not aug and self.rotate != 0:
            transform_list = ['Scale', 'RandomRotation', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class Metric:

    def __init__(self):
        self.rights = 0
        self.wrongs = 0

    def update_acc(self, output, label):
        pred = (output >= 0)
        batch_rights = sum(label.type(torch.int64) == pred.type(torch.int64)).cpu().numpy()[0]

        self.rights += batch_rights
        self.wrongs += (label.shape[0] - batch_rights)

    def get_acc(self):
        print('rights: ', self.rights)
        print('wrongs: ', self.wrongs)
        return ((self.rights) / (self.rights + self.wrongs)) * 100

    def reset_acc(self):
        self.rights = 0
        self.wrongs = 0


# '../../dataset/omniglot/python/images_background'
# '../../dataset/omniglot/python/images_evaluation'
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"

    parser.add_argument('-dsn', '--dataset_name', default='omniglot', choices=['omniglot', 'cub', 'hotels'])
    parser.add_argument('-dsp', '--dataset_path', default='CUB_200_2011/')
    parser.add_argument('-sdp', '--subdir_path', default='images/')
    parser.add_argument('-trp', '--train_path', default='./omniglot/python/images_background')
    parser.add_argument('-tsp', '--test_path', default='./omniglot/python/images_evaluation')
    parser.add_argument('-sp', '--save_path', default='models/', help="path to store model")
    parser.add_argument('-a', '--aug', default=False, action='store_true')
    parser.add_argument('-r', '--rotate', default=0.0, type=float, help='store_true')

    parser.add_argument('-s', '--seed', default=402, type=int, help="random seed")
    parser.add_argument('-w', '--way', default=20, type=int, help="how much way one-shot learning")
    parser.add_argument('-t', '--times', default=400, type=int, help="number of samples to test accuracy")
    parser.add_argument('-wr', '--workers', default=4, type=int, help="number of dataLoader workers")
    parser.add_argument('-bs', '--batch_size', default=128, type=int, help="number of batch size")
    parser.add_argument('-lr', '--lr', default=0.00006, type=float, help="learning rate")
    parser.add_argument('-lf', '--log_freq', default=10, type=int, help="show result after each show_every iter.")
    parser.add_argument('-sf', '--save_freq', default=100, type=int, help="save model after each save_every iter.")
    parser.add_argument('-tf', '--test_freq', default=100, type=int, help="test model after each test_every iter.")
    # parser.add_argument('-ms', '--max_steps', default=50000, type=int, help="number of steps before stopping")
    parser.add_argument('-ep', '--epochs', default=1, type=int, help="number of epochs before stopping")

    parser.add_argument('-1cf', '--first_conv_filter', default=10, type=int, help="")
    parser.add_argument('-2cf', '--second_conv_filter', default=7, type=int, help="")
    parser.add_argument('-3cf', '--third_conv_filter', default=4, type=int, help="")
    parser.add_argument('-4cf', '--fourth_conv_filter', default=4, type=int, help="")
    parser.add_argument('-5cf', '--fifth_conv_filter', default=3, type=int, help="")
    parser.add_argument('-6cf', '--sixth_conv_filter', default=3, type=int, help="")
    parser.add_argument('-7cf', '--seventh_conv_filter', default=3, type=int, help="")
    parser.add_argument('-co', '--conv_output', default=2304, type=int, help="")
    parser.add_argument('-ll', '--last_layer', default=4096, type=int, help="number of last layer neurons.")

    args = parser.parse_args()

    return args


def test_model(net, args, dataLoader):
    right, error = 0, 0
    for _, (test1, test2) in enumerate(dataLoader, 1):
        if args.cuda:
            test1, test2 = test1.cuda(), test2.cuda()
        test1, test2 = Variable(test1), Variable(test2)
        output = net.forward(test1, test2).data.cpu().numpy()
        pred = np.argmax(output)
        if pred == 0:
            right += 1
        else:
            error += 1

    return

    print('*' * 70)
    print('epoch: %d, batch: [%d]\tTest set\tcorrect:\t%d\terror:\t%d\taccuracy:\t%f' % (
        epoch, total_batch_id, right, error, right * 1.0 / (right + error)))
    print('*' * 70)
    queue.append(right * 1.0 / (right + error))
