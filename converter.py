import sys
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from neural.net import *
from neural.train import *
from tools.dataloader import *
from tools.transform import *

def usage():
    print("Voici les commandes disponibles : ")
    print("  ~$ resize width height option=<value>")
    print("\t - path : the dataset to resize, otherwise resize \"data/pictures/\"")
    print("\t - out : the output folder for the resized images")
    print("  ~$ train option=<value> . . .")
    print("    * si type ou path n'est pas defini un model lineaire sera cree *")
    print("    * si l'option save n'est pas defini le model sera sauve dans le fichier \"model\" *")
    print("\t - type : create a new neural net with the desired model (linear|conv|unet)")
    print("\t - model : trains an existing model, type should be specified in this case")
    print("\t - save : the path to save the model after training")
    print("\t - epoch : the number of epochs to train the model")
    print("\t - res : the resolution of the pictures (ex:res=32x32)")
    print("\t - trainset : the path to the training data, the res option should be defined")
    print("\t              and data should be organized as so:")
    print("\t\t trainset")
    print("\t\t    └ mini-batch0")
    print("\t\t        └ pic0")
    print("\t\t        ├ pic1")
    print("\t\t        ├ ...")
    print("\t\t    └ mini-batch1")
    print("\t\t        └ pic0")
    print("\t\t        ├ pic1")
    print("\t\t        ├ ...")
    print("\t\t    └ ...")
    print("\t - cuda : a boolean to enable or disable GPU (ex: cuda=True | cuda=False)")
    print("  ~$ convert path/to/file.png (a complete folder could be given too)")


def training(type, img_res=IMG_RES, img_path=IMG_PATH, epoch=1, model=None, save="model/net.pth", cuda=True):
    ultra_net = UltraNet(type, img_size, model)
    net = ultra_net.get_net()

    trainset = DataLoader(img_path, transform=True, dim=img_res)
    train(net, epoch, trainset, cuda=cuda)

    torch.save(net.state_dict(), save)

def resize(w, h, img=IMG_PATH, out=None):
    size = [w, h]
    data = DataLoader(img, transform=True)
    data.save_resize(size[0], size[1], out)
    s = data.get_img_size()

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 2 :
        usage()
        exit()

    if sys.argv[1] == "help":
        usage()

    elif sys.argv[1] == "resize":
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
                    IMG_PATH = val

                if opt == "out":
                    out = val

            resize(w,h, IMG_PATH, out)

    elif sys.argv[1] == "convert" :
        pass

    elif sys.argv[1] == "train" :
        if argc < 2 :
            usage()
            exit()
        else :
            IMG_RES[0], IMG_RES[1] = 32, 32
            IMG_PATH = "data/pictures_" + str(IMG_RES[0]) + "x" + str(IMG_RES[1]) + "/"

            net_type = "linear"
            model = None
            epoch = 1
            save = "model/net.pth"
            cuda = True

            options = sys.argv[2:]
            for option in options:
                opt, val = option.split("=")[0], option.split("=")[1]

                if opt == "epoch" :
                    epoch = int(val)

                elif opt == "save" :
                    save = val

                elif opt == "type" :
                    net_type = val

                elif opt == "model" :
                    model = val

                elif opt == "trainset" :
                    IMG_PATH = val

                elif opt == "cuda" :
                    cuda = bool(val)

                elif opt == "res" :
                    try :
                        width, height = val.split("x")[0], val.split("x")[1]
                        IMG_RES[0], IMG_RES[1] = int(width), int(height)
                    except Exception as e :
                        print(e)
                        raise e

            print("\nPARAMS : ")
            print("  RES", IMG_RES)
            print("  PATH", IMG_PATH)
            print("  EPOCH", epoch)
            print("  SAVE", save)
            print("  CUDA", cuda)
            print("")

            training(type, epoch=epoch, model=model, save=save, cuda=cuda)
