import sys
import numpy
import sklearn.metrics
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

def to_image(arr, mode):
    shape = (IMG_RES[0], IMG_RES[1])
    if (mode == "RGB"):
        shape += (3,)
    arr = torch.reshape(arr, shape).cpu().detach()
    arr = numpy.asarray(arr)
    img = Image.fromarray(arr, mode)
    return img

def view_data(arr, mode):
    shape = (IMG_RES[0], IMG_RES[1])
    if (mode == "RGB"):
        shape += (3,)
    arr = torch.reshape(arr, shape).cpu().detach()
    arr = numpy.asarray(arr)
    img = Image.fromarray(arr, mode)
    return img.show()

def train(net, epochs, trainloader, cuda=False):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss(reduction="sum")
    # criterion = nn.KLDivLoss()
    # criterion = nn.NLLLoss()

    # optimizer = optim.ASGD(net.parameters(), lr=0.00001, lambd=0.0001, alpha=0.5)
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    if cuda:
        criterion.cuda()
        net.cuda()

    e_losses = []
    for epoch in range(epochs): # loop over the dataset multiple times

        losses = 0
        running_loss = 0.0
        for i, item in enumerate(trainloader, 0):

            # UNPACKING item
            path = item["path"]
            data = item["data"]
            label = item["label"]

            # REFORMAT DATA AND LABEL
            data = torch.tensor(data, dtype=torch.float32)
            data = data.reshape(-1)
            label = torch.tensor(label, dtype=torch.float32)
            # label = label.reshape(-1)

            # print(path, ": data", data.shape[0], "| label", label.shape[0])

            if cuda:
                # LOADS MODEL ON GPU
                device = torch.device('cuda:0')
                data = data.to(device)
                label = label.to(device)

            optimizer.zero_grad()

            out = net(data)
            out = torch.reshape(out, (IMG_RES[0], IMG_RES[1], 3))

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses += running_loss

            if i % 300 == 299:    # print every 300 mini-batches
                print('[e: %d, i: %d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                # running_loss = 0.0


        e_losses.append(losses / len(trainloader))
        print("\n\tAVG, e=%d: %.3f\n" % (epoch+1, losses / len(trainloader)))
        print("\n#######################################################################\n")

    plt.plot(e_losses, label="epoch (avg) running loss")
    plt.show()

def test(net, testloader, cuda=False):
    net.eval()
    if cuda:
        net.cuda()

    accuracy, jaccard, mse, maxe = 0, 0, 0, 0
    for i, item in enumerate(testloader, 0):
        if item is None:
            break
        # UNPACKING item
        path = item["path"]
        data = item["data"]
        label = item["label"]

        # REFORMAT DATA AND LABEL
        data = torch.tensor(data, dtype=torch.float32)
        data = data.reshape(-1)
        label = torch.tensor(label, dtype=torch.float32)
        label = label.reshape(-1)

        if cuda:
            # LOADS MODEL ON GPU
            device = torch.device('cuda:0')
            data = data.to(device)
            label = label.to(device)

        out = net(data)

        label = label.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        # accuracy += sklearn.metrics.accuracy_score(label, out)
        # jaccard  += sklearn.metrics.jaccard_score(label, out)
        mse += sklearn.metrics.mean_squared_error(label, out)
        maxe += sklearn.metrics.max_error(label, out)

    # accuracy = accuracy / len(testloader)
    # jaccard = jaccard / len(testloader)
    mse = mse / len(testloader)
    maxe = maxe / len(testloader)
    # print("Accuracy ==> [%d] on %d images" % (accuracy*100, len(testloader)))
    # print("Jaccard ==> [%d] on %d images" % (jaccard*100, len(testloader)))
    print("MSE ==> [%.5f] on %d images" % (mse, len(testloader)))
    print("MAX_ERROR ==> [%.5f] on %d images" % (maxe, len(testloader)))

def convert(net, input, view=False):
    # img should be a flatten tensor
    out = net(input)
    img = to_image(out, "RGB")

    if view:
        view_data(input, 'L')
        view_data(out, "RGB")

    return img


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
