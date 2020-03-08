import sys
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

import neural.net as nnet
import neural.train as train
import tools.dataloader as dataloader

def usage():
    print("Voici les commandes disponibles : ")
    print("  ~$ resize width height option=<value>")
    print("\t - path : the dataset to resize, otherwise resize \"data/pictures/\"")
    print("\t - out : the output folder for the resized images")
    print("  ~$ train res=48x48 trainset=data/paysage_48x48/ option=<value> . . .")
    print("\t - trainset : the path to the training data, the res option should be defined, default : \"data/paysage_48x48/\"")
    print("\t- res : the resolution of the pictures (ex:res=32x32), default : 48x48")
    print("\t - save : the path to save the model after training, default : \"model/net.pth\"")
    print("\t - model : trains an existing model")
    print("\t - epoch : the number of epochs to train the model")
    print("\t - cuda : a boolean to enable or disable GPU (ex: cuda=True | cuda=False)")
    print("  ~$ convert path/to/file.png (a complete folder could be given too) option=<value> ")
    print("\t - model : the model to load to convert your picture(s)")
    print("\t - save : the path to save the converted pictures, otherwise saves them in the same folder as the original")
    print("\t - view : show the result before saving")


def training(img_res=dataloader.IMG_RES, img_path=dataloader.IMG_PATH, epoch=1, model=None, save="model/net.pth", cuda=True):
    print("begin process...")
    # net = ultra_net.get_net()
    # ultra_net = nnet.UltraNet(type, img_res, model) #loading model

    net = nnet.NetConv(img_res)
    if model is not None:
        net.load_state_dict(torch.load(path))

    dataset = dataloader.Dataloader(img_path, test_per=10, mode="train") #dataset
    print("train : %d pictures" % (len(dataset)))
    # trains the model, plotting the loss over the epochs at the end
    train.train(net, epoch, dataset, cuda)

    dataset.set_mode("test")
    print("test : %d pictures" % (len(dataset)))
    train.test(net, dataset, cuda) # test the model accuracy

    torch.save(net.state_dict(), save) #save the model in save path parameter

def resize(w, h, img_path=dataloader.IMG_PATH, out=None):
    print("begin process...")
    size = [w, h]
    data = dataloader.Dataloader(img_path) #the data to be resized

    if out is None:
        out = img_path[:len(img_path)-1] + '_' + str(w) + 'x' + str(h) + '/'
        try:
            os.mkdir(out)
        except Exception as e:
            pass
    for i in range(len(data)):
        filename = data[i]

        img = data.resize_image(img_path + filename, (w,h))
        img.save(out + filename, "PNG")

        if i%100 == 99:
            print("[%d / %d] %s" % (i, len(data), filename))

def convert(path, model="model/net.pth", save=None, view=False):
    print("begin process...")
    net = nnet.NetConv(dataloader.IMG_RES)
    if model is not None:
        net.load_state_dict(torch.load(model))

    imgs = dataloader.Dataloader(path, transform=True, dim=dataloader.IMG_RES) #dataset
    for i, item in enumerate(imgs):
        data = item["data"]
        data = torch.tensor(data, dtype=torch.float32)
        data = data.reshape(-1)
        img = train.convert(net, data, view)
        break

    if save is not None:
        img.save(save, "PNG")


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 2 :
        usage()
        exit()

    if sys.argv[1] == "help":
        usage()

    elif sys.argv[1] == "resize":
        print("resize... \n" + str(sys.argv))
        if argc < 4 :
            usage()
            exit()
        else :
            w,h = None, None
            out = None
            try :
                w, h = int(sys.argv[2]), int(sys.argv[3])
            except Exception as e:
                print(e)
                raise e

            options = sys.argv[4:]
            for option in options:
                opt, val = option.split("=")[0], option.split("=")[1]

                if opt == "path":
                    dataloader.IMG_PATH = val

                if opt == "out":
                    out = val

            resize(w, h, dataloader.IMG_PATH, out)

    elif sys.argv[1] == "convert" :
        print("convert... \n" + str(sys.argv))
        if argc < 3 :
            usage()
            exit()
        else :

            path = sys.argv[2]
            model = "model/net.pth"
            save = None
            view = False

            options = sys.argv[3:]
            for option in options:
                opt, val = option.split("=")[0], option.split("=")[1]

                if opt == "model":
                    model = val
                if opt == "save":
                    save = val
                if opt == "view":
                    view = bool(val)

            convert(path, model, save, view)

    elif sys.argv[1] == "train" :
        print("training... \n" + str(sys.argv))

        if argc < 2 :
            usage()
            exit()
        else :

            model = None
            e = 1
            save = "model/net.pth"
            cuda = True

            options = sys.argv[2:]
            for option in options:
                opt, val = option.split("=")[0], option.split("=")[1]
                print(opt, val)
                if opt == "epoch" :
                    e = int(val)

                if opt == "save" :
                    save = val

                if opt == "model" :
                    model = val

                if opt == "trainset" :
                    dataloader.IMG_PATH = val

                if opt == "cuda" :
                    if (val.lower() == "true"):
                        cuda = True
                    elif (val.lower() == "false"):
                        cuda = False

                if opt == "res" :
                    try :
                        width, height = val.split("x")[0], val.split("x")[1]
                        dataloader.IMG_RES[0], dataloader.IMG_RES[1] = int(width), int(height)
                    except Exception as e :
                        print(e)
                        raise e

            print("\nPARAMS : ")
            print("  RES", dataloader.IMG_RES)
            print("  PATH", dataloader.IMG_PATH)
            print("  EPOCH", e)
            print("  MODEL", model)
            print("  SAVE", save)
            print("  CUDA", cuda)
            print("")

            training(dataloader.IMG_RES, dataloader.IMG_PATH, epoch=e, model=model, save=save, cuda=cuda)


    print("done...")
