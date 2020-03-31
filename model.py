import torch
import torch.nn as nn


class Siamese(nn.Module):

    def __init__(self, args):
        super(Siamese, self).__init__()
        if args.dataset_name == 'cub': #  84 * 84
            input_channel = 3
            # last_layer = 2304
            last_layer = 9216
        elif args.dataset_name == 'omniglot':
            input_channel = 1
            last_layer = 9216
        else:
            raise Exception('Dataset not supported')

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 64, 10),  # 64@491*491
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@246*246

            nn.Conv2d(64, 128, 7),
            nn.ReLU(),  # 128@240*240
            nn.MaxPool2d(2),  # 128@120*120

            nn.Conv2d(128, 128, 4),  # 128@117*117
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128@59*59

            nn.Conv2d(128, 256, 4),
            nn.ReLU(),  # 256@56*56
            # nn.MaxPool2d(2),  # 256@28*28

            # nn.Conv2d(256, 512, 4),
            # nn.ReLU(),  # 512@25*25
            # nn.MaxPool2d(2),  # 512@13*13
            #
            # nn.Conv2d(512, 512, 4),
            # nn.ReLU(),  # 512@10*10
            # nn.MaxPool2d(2),  # 512@5*5
        )
        self.linear = nn.Sequential(nn.Linear(last_layer, 4096), nn.Sigmoid())  #TODO
        # self.linear = nn.Sequential(nn.Linear(8192, 4096), nn.Sigmoid())  # 512 * 4 * 4 input
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out

#
# # for test
# if __name__ == '__main__':
#     net = Siamese()
#     print(net)
#     print(list(net.parameters()))
