import os
import pickle
import random
import time
from collections import deque

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from model import Siamese
from mydataset import OmniglotTrain, OmniglotTest, CUBTrain, CUBTest

if __name__ == '__main__':

    args = utils.get_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    # train_dataset = dset.ImageFolder(root=Flags.train_path)
    # test_dataset = dset.ImageFolder(root=Flags.test_path)

    if args.gpu_ids != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print("use gpu:", args.gpu_ids, "to train.")

    trainSet = None
    testSet = None

    if args.dataset_name == 'cub':
        trainSet = CUBTrain(args, transform=data_transforms)
        testSet = CUBTest(args, transform=transforms.ToTensor())
    elif args.dataset_name == 'omniglot':
        trainSet = OmniglotTrain(args, transform=data_transforms)
        testSet = OmniglotTest(args, transform=transforms.ToTensor())
    else:
        print('Fuck: ', args.dataset_name)

    print(args.way)

    testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)

    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    net = Siamese(args)

    # multi gpu
    if len(args.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if args.cuda:
        net.cuda()

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    opt.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque()

    print('steps:', args.max_steps)

    epochs = int(args.max_steps / len(trainLoader))

    total_batch_id = 0
    metric = utils.Metric()
    for epoch in range(epochs):

        for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
            if batch_id > args.max_steps:
                break
            if args.cuda:
                img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
            else:
                img1, img2, label = Variable(img1), Variable(img2), Variable(label)
            opt.zero_grad()
            output = net.forward(img1, img2)
            metric.update_acc(output, label)
            loss = loss_fn(output, label)
            loss_val += loss.item()
            loss.backward()

            opt.step()
            total_batch_id += 1

            if total_batch_id % 100 == 0:
              print('batch: ', total_batch_id)

            if total_batch_id % args.log_freq == 0:
                print('epoch: %d, batch: [%d]\tacc:\t%.5f\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
                    epoch, total_batch_id, metric.get_acc(), loss_val / args.log_freq, time.time() - time_start))
                loss_val = 0
                metric.reset_acc()
                time_start = time.time()

            if total_batch_id % args.save_freq == 0:
                torch.save(net.state_dict(), args.save_path + '/model-inter-' + str(batch_id + 1) + ".pt")

            if total_batch_id % args.test_freq == 0:
                right, error = 0, 0
                for _, (test1, test2) in enumerate(testLoader, 1):
                    if args.cuda:
                        test1, test2 = test1.cuda(), test2.cuda()
                    test1, test2 = Variable(test1), Variable(test2)
                    output = net.forward(test1, test2).data.cpu().numpy()
                    pred = np.argmax(output)
                    if pred == 0:
                        right += 1
                    else:
                        error += 1

                print('*' * 70)
                print('epoch: %d, batch: [%d]\tTest set\tcorrect:\t%d\terror:\t%d\taccuracy:\t%f' % (
                    epoch, total_batch_id, right, error, right * 1.0 / (right + error)))
                print('*' * 70)
                queue.append(right * 1.0 / (right + error))
            train_loss.append(loss_val)
        #  learning_rate = learning_rate * 0.95

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#" * 70)
    print('queue len: ', len(queue))
    print("final accuracy: ", acc / len(queue))
