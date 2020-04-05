import logging
import pickle
import sys
import time
from collections import deque

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import utils
from dataloader import *
from model import Siamese


def _logger():
    logging.basicConfig(format='%(message)s', stream=sys.stdout, level=logging.INFO)
    return logging.getLogger()


def main():
    logger = _logger()

    args = utils.get_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_size = -1

    if args.dataset_name == 'cub':
        image_size = 84
    elif args.dataset_name == 'omniglot':
        image_size = 105
    elif args.dataset_name == 'hotels':
        image_size = 300

    data_transforms = utils.TransformLoader(image_size, rotate=args.rotate).get_composed_transform(aug=args.aug)

    # data_transforms = transforms.Compose([
    #     transforms.Resize([int(image_size), int(image_size)]),
    #     transforms.RandomAffine(15),
    #     transforms.ToTensor()
    # ])

    # train_dataset = dset.ImageFolder(root=Flags.train_path)
    # test_dataset = dset.ImageFolder(root=Flags.test_path)

    if args.gpu_ids != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print("use gpu:", args.gpu_ids, "to train.")

    trainSet = None
    testSet = None
    valSet = None

    if args.dataset_name == 'cub':
        trainSet = CUBTrain(args, transform=data_transforms)
        valSet = CUBTest(args, transform=data_transforms, mode='val')
        testSet = CUBTest(args, transform=data_transforms)
    elif args.dataset_name == 'omniglot':
        trainSet = OmniglotTrain(args, transform=data_transforms)
        # valSet = CUBTest(args, transform=data_transforms, mode='val')
        testSet = OmniglotTest(args, transform=transforms.ToTensor())
    elif args.dataset_name == 'hotels':
        trainSet = HotelTrain(args, transform=data_transforms)
        # valSet = CUBTest(args, transform=data_transforms, mode='val')
        testSet = HotelTest(args, transform=data_transforms)
    else:
        print('Fuck: ', args.dataset_name)

    print('way:', args.way)

    testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)

    if valSet is not None:
        valLoader = DataLoader(valSet, batch_size=args.way, shuffle=False, num_workers=args.workers)

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

    train_losses = []
    time_start = time.time()
    queue = deque(maxlen=20)

    # print('steps:', args.max_steps)

    # epochs = int(np.ceil(args.max_steps / len(trainLoader)))
    epochs = args.epochs
    print('epochs: ', epochs)

    total_batch_id = 0
    metric = utils.Metric()

    max_val_acc = 0
    best_model = ''

    for epoch in range(epochs):

        train_loss = 0
        metric.reset_acc()

        with tqdm(total=len(trainLoader), desc=f'Epoch {epoch + 1}/{args.epochs}') as t:
            for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
                if args.cuda:
                    img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
                else:
                    img1, img2, label = Variable(img1), Variable(img2), Variable(label)

                net.train()
                opt.zero_grad()

                output = net.forward(img1, img2)
                metric.update_acc(output, label)
                loss = loss_fn(output, label)

                train_loss += loss.item()
                loss.backward()

                opt.step()
                total_batch_id += 1
                t.set_postfix(loss=f'{train_loss / batch_id:.4f}', train_acc=f'{metric.get_acc():.4f}')

                # if total_batch_id % args.log_freq == 0:
                #     logger.info('epoch: %d, batch: [%d]\tacc:\t%.5f\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
                #         epoch, batch_id, metric.get_acc(), train_loss / args.log_freq, time.time() - time_start))
                #     train_loss = 0
                #     metric.reset_acc()
                #     time_start = time.time()

                if total_batch_id % args.test_freq == 0:
                    net.eval()
                    right, error = 0, 0
                    val_label = np.zeros(shape=args.way, dtype=np.float32)
                    val_label[0] = 1
                    val_label = torch.from_numpy(val_label).reshape((args.way, 1))

                    if args.cuda:
                        val_label = Variable(val_label.cuda())
                    else:
                        val_label = Variable(val_label)

                    if valSet is not None:
                        loader = valLoader
                    else:
                        loader = testLoader

                    for _, (test1, test2) in enumerate(loader, 1):
                        if args.cuda:
                            test1, test2 = test1.cuda(), test2.cuda()
                        test1, test2 = Variable(test1), Variable(test2)
                        output = net.forward(test1, test2)
                        val_loss = loss_fn(output, val_label)
                        output = output.data.cpu().numpy()
                        pred = np.argmax(output)
                        if pred == 0:
                            right += 1
                        else:
                            error += 1

                    val_acc = right * 1.0 / (right + error)
                    logger.info('*' * 70)

                    logger.info(
                        'epoch: %d, batch: [%d]\tVal set\tcorrect:\t%d\terror:\t%d\tval_acc:%f\tval_loss:\t%f' % (
                            epoch, batch_id, right, error, val_acc, val_loss))
                    logger.info('*' * 70)

                    if val_acc > max_val_acc:
                        logger.info(
                            'saving model... current val acc: [%f], previous val acc [%f]' % (val_acc, max_val_acc))
                        max_val_acc = val_acc
                        best_model = 'model-inter-' + str(total_batch_id + 1) + '-epoch-' + str(
                            epoch + 1) + '-val-acc-' + str(val_acc) + '.pt'
                        torch.save({'epoch': epoch,
                                    'model_state_dict': net.state_dict()},
                                   args.save_path + '/model-inter-' + str(total_batch_id + 1) + 'val-acc-' + str(
                                       val_acc) + '.pt')
                    else:
                        logger.info('Not saving, best val [%f], current was [%f]' % (max_val_acc, val_acc))

                    queue.append(right * 1.0 / (right + error))
                train_losses.append(train_loss)

                t.update()

    # testing
    tests_right, tests_error = 0, 0

    checkpoint = torch.load(best_model)
    logger.info('Loading model from epoch [%d]' % checkpoint['epoch'])
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()

    for _, (test1, test2) in enumerate(testLoader, 1):
        if args.cuda:
            test1, test2 = test1.cuda(), test2.cuda()
        test1, test2 = Variable(test1), Variable(test2)
        output = net.forward(test1, test2).data.cpu().numpy()
        pred = np.argmax(output)
        if pred == 0:
            tests_right += 1
        else:
            tests_error += 1

    test_acc = tests_right * 1.0 / (tests_right + tests_error)
    logger.info('$' * 70)
    logger.info(
        'TEST:\tTest set\tcorrect:\t%d\terror:\t%d\ttest_acc:%f\t' % (tests_right, tests_error, test_acc))
    logger.info('$' * 70)

    #  learning_rate = learning_rate * 0.95

    with open('train_losses', 'wb') as f:
        pickle.dump(train_losses, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#" * 70)
    print('queue len: ', len(queue))
    print("final accuracy: ", acc / len(queue))


if __name__ == '__main__':
    main()
