import torch
import torch.nn as nn
import torch.nn.functional as F

# IMG_RES = [153, 102]

# class UltraNet(object) :
#     def __init__(self, type, img_size, path=None):
#         self.type = type
#         self.net = None
#         if self.type == "linear" :
#             self.net = NetLinear(img_size)
#         elif type == "conv":
#             self.net = NetConv(img_size)
#         elif type == "unet":
#             self.net = NetU(img_size)
#
#         if path is not None:
#             self.net.load_state_dict(torch.load(path))
#
#     def get_net(self):
#         return self.net

class NetLinear(nn.Module): # NO RESULTS
    def __init__(self, img_size):
        super(NetLinear, self).__init__()
        print("Net(res:" + str(img_size) + ") : " + str(img_size[0]*img_size[1]) + " --> " + str(img_size[0]*img_size[1]*3))

        self.w, self.h = img_size[0], img_size[1]
        self.dim = img_size[0]*img_size[1]

        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim*2)
        self.layer3 = nn.Linear(self.dim*2, self.dim*2)
        self.layer4 = nn.Linear(self.dim*2, self.dim*3)
        self.layer5 = nn.Linear(self.dim*3, self.dim*3)
        self.layer6 = nn.Linear(self.dim*3, self.dim*3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x

class NetConv(nn.Module):
    def __init__(self, img_size):
        super(NetConv, self).__init__()
        print("Net(res:" + str(img_size) + ") : " + str(img_size[0]*img_size[1]) + " --> " + str(img_size[0]*img_size[1]*3))

        self.w, self.h = img_size[0], img_size[1]
        self.dim = img_size[0]*img_size[1]

        self.in_linear = nn.Linear(self.dim, self.dim)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=2, stride=1, padding=0)
        self.out_linear = nn.Linear(3 * self.dim, 3 * self.dim)

    def forward(self, x):
        x = F.relu(self.in_linear(x))
        x = x.view(1, 1, self.w, self.h)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(3 * self.dim)
        x = F.relu(self.out_linear(x))
        return x

    def get_dim(self):
        return self.dim

class NetU(nn.Module):
    def __init__(self, img_size):
        super(NetU, self).__init__()
        print("Net(res:" + str(img_size) + ") : " + str(img_size[0]*img_size[1]) + " --> " + str(img_size[0]*img_size[1]*3))

    def forward(self, x):
        return x

if __name__ == "__main__":
    torch.cuda.set_device(0)
    net = Net([32,32])
    net.cuda()
