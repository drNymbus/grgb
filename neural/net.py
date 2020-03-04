import torch
import torch.nn as nn
import torch.nn.functional as F

# IMG_RES = [153, 102]

class UltraNet(object) :
    def __init__(self, type, img_size, path=None):
        self.type = type
        self.net = None
        if type == "linear" :
            self.net = NetLinear(img_size)
        elif type == "conv":
            self.net = NetLinear(img_size)
        elif type == "unet":
            self.net = NetLinear(img_size)

        if path is not None:
            self.net.load_state_dict(torch.load(path))

    def get_net():
        return self.net

class NetLinear(nn.Module):
    def __init__(self, img_size):
        super(Net, self).__init__()
        print("net size", img_size)

        dim = img_size[0]*img_size[1]
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim*2)
        self.layer3 = nn.Linear(dim*2, dim*3)
        self.layer4 = nn.Linear(dim*3, dim*3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x

if __name__ == "__main__":
    torch.cuda.set_device(0)
    net = Net([32,32])
    net.cuda()
