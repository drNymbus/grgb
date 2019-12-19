import sys
# import numpy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from PIL import Image

from neural.net import Net
from gatherer.dataloader import DataLoader
from gatherer.transform import DataLoader


def usage():
    print("Bla bla bla")
    print("Bla bla bla")
    print("Bla bla bla")

if __name__ == "__main__":
    if sys.argc < 2:
        usage()

    IMG_RES[0], IMG_RES[1] = 64, 64
    print("train size", IMG_RES)

    net = Net(IMG_RES)

    trainset = DataLoader(IMG_PATH, transform=True, dim=IMG_RES)
    # trainset = [trainset[0]]
    # print(trainset)

    train(net, 1, trainset, cuda=True)
