import sys
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

from neural.net import *
from tools.dataloader import *

# IMG_RES = [153, 102]
# IMG_PATH = "data/"
# JSON_PATH = "data/json/"

def view_data(arr, mode):
    shape = (IMG_RES[0], IMG_RES[1])
    if (mode == "RGB"):
        shape += (3,)
    arr = torch.reshape(arr, shape).cpu().detach()
    arr = numpy.asarray(arr)
    img = Image.fromarray(arr, mode)
    return img.show()

def train(net, epochs, trainloader, cuda=False):
    if cuda:
        net.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.ASGD(net.parameters(), lr=0.0001)
    # optimizer = optim.Adam(net.parameters(), lr=0.0000000001)

    for epoch in range(epochs):  # loop over the dataset multiple times

        losses = []
        running_loss = 0.0
        for i, item in enumerate(trainloader, 0):

            if item is None:
                break
            # UNPACKING item
            # path, data, label, index = item["path"], item["data"], item["label"], item["index"]
            path = item["path"]
            data = item["data"]
            label = item["label"]
            index = item["index"]

            data = torch.tensor(data, dtype=torch.float32)
            data = data.reshape(-1)

            label = torch.tensor(label, dtype=torch.float32)
            label = label.reshape(-1)

            # LOADS MODEL ON GPU
            if cuda:
                device = torch.device('cuda:0')
                data = data.to(device)
                label = label.to(device)

            # print("data", type(data), data.size())
            # print("label", type(label), label.size())

            optimizer.zero_grad()

            out = net(data)

            loss = criterion(out, label)
            # print("LOSS: " + str(loss))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(running_loss)
            if i % 200 == 199:    # print every 2000 mini-batches
                # print(type(out), out.size())
                print('[e: %d, i: %d] (mini-batch: %d, mini-i: %d) loss: %.3f' %
                      (epoch + 1, i + 1, index["dir"], index["file"], running_loss / 50))
                running_loss = 0.0
                # view_data(data, 'L')
                # view_data(out, "RGB")

        plt.plot(losses)
        plt.ylabel("Loss")
        plt.show()

if __name__ == "__main__":
    # IMG_RES[0], IMG_RES[1] = 64, 64
    # print("train size", IMG_RES)
    #
    # net = Net(IMG_RES)
    #
    # trainset = DataLoader(IMG_PATH, transform=True, dim=IMG_RES)
    # # trainset = [trainset[0]]
    # # print(trainset)
    #
    # train(net, 1, trainset, cuda=True)
    pass
