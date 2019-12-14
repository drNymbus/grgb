import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_RES = [153, 102]

class LinearNet(nn.Module):
    def __init__(self, img_size):
        global IMG_RES
        super(Net, self).__init__()
        IMG_RES[0], IMG_RES[1] = img_size[0], img_size[1]
        print("net size", IMG_RES)

        dim = IMG_RES[0]*IMG_RES[1]
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim*3)
        self.layer3 = nn.Linear(dim*3, dim*3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

if __name__ == "__main__":
    torch.cuda.set_device(0)
    net = Net([32,32])
    net.cuda()
