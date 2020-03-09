import sys
import json
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

def train(net, epochs, trainloader, cuda=False, plot=False):
    JSON_PATH = "data/json/loss.json"
    with open(JSON_PATH, "w+") as f:
        json.dump({}, f)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss()
    # criterion = nn.NLLLoss()

    # optimizer = optim.ASGD(net.parameters(), lr=0.0000001)
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    if cuda:
        criterion.cuda()
        net.cuda()

    e_losses = []
    for epoch in range(epochs): # loop over the dataset multiple times

        losses = []
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
            losses.append(running_loss)

            if i % 300 == 299:    # print every 300 mini-batches
                print('[e: %d, i: %d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                # running_loss = 0.0

        if plot:
            plt.plot(losses, label="running loss for epoch(%d)"%(epoch+1))
            plt.show()

        avg = 0 # compute average loss over the epoch
        for loss in losses:
            avg += loss
        avg = avg / len(trainloader)
        e_losses.append(avg)

        # loads data in data/json/loss.json
        all_data = {}
        try:
            with open(JSON_PATH, 'r') as f:
                all_data = json.loads(f.read())
        except Exception as e:
            print("/!\\ Warning /!\\ " + str(e))
            pass

        # append the loss data of the epoch in data/json/loss.json
        all_data[epoch+1] = {"average" : avg, "losses" : losses}
        with open(JSON_PATH, "w+") as f:
            json.dump(all_data, f)

        print("\n\t(AVERAGE) e=%d: loss=%.3f\n" % (epoch+1, avg))
        print("#########################################################")

    if plot:
        plt.plot(e_losses, label="epoch (average) running loss")
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
    print("MEAN_SQUARED_ERROR ==> [%.5f] on %d images" % (mse, len(testloader)))
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
