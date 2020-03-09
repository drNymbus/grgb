import sys
import os
import numpy
import random
from PIL import Image

from torch.utils.data import Dataset

IMG_RES = [48,48]
IMG_PATH = None
IMG_PATH_PAYSAGE = "data/paysage_48x48/"
IMG_PATH_NATURE = "data/nature_48x48/"

class Dataloader(object):
    def __init__(self, path, test_per=0, mode=None):#, gray=False):
        self.path = path
        self.files = []
        for _,_,files in os.walk(self.path):
            for file in files:
                self.files.append(file)
        # self.files = self.files[:len(self.files)//150]

        self.transform = Transform()
        # self.gray = gray # determine if pictures are in grayscale or not
        self.mode = mode

        random.shuffle(self.files)
        i = test_per * len(self.files) // 100
        self.trainset= self.files[i:]
        self.testset = self.files[:i]

    def get_path(self):
        return self.path

    def set_mode(self, mode):
        if (mode == "train" or mode == "test"):
            self.mode = mode
        else:
            self.mode = "none"

    def __len__(self):
        if self.mode == "train":
            return len(self.trainset)
        if self.mode == "test":
            return len(self.testset)
        else:
            return len(self.files)

    def load_image(self, path):
        ''' return an image Image() '''
        img = Image.open(path)
        return img

    def resize_image(self, path, size):
        ''' return the image at path resized with size(w,h) '''
        img = self.load_image(path)
        img = self.transform.resize(img, size)
        return img

    def __getitem__(self, i):
        if self.mode == "train":
            path = self.path + self.trainset[i] if self.path is not None else self.trainset[i]
            img = self.load_image(path)
            gray = self.transform.grayscale(img)

            item = {}
            item["path"] = self.path + self.trainset[i] if self.path is not None else self.trainset[i]
            item["data"] = numpy.asarray(gray)
            item["label"] = numpy.asarray(img)
            return item

        elif self.mode == "test":
            path = self.path + self.testset[i] if self.path is not None else self.testset[i]
            img = self.load_image(path)
            gray = self.transform.grayscale(img)

            item = {}
            item["path"] = self.path + self.trainset[i] if self.path is not None else self.trainset[i]
            item["data"] = numpy.asarray(gray)
            item["label"] = numpy.asarray(img)
            return item

        else:
            return self.files[i]

    def merge(self, data, test_per=0):
        self.set_mode("none")
        data.set_mode("none")
        for i in range(len(self)):
            p = self.path + self.files[i]
            self.files[i] = p
        self.path = None

        for i in range(len(data)):
            p = data.get_path() + data[i]
            self.files.append(p)

        random.shuffle(self.files)
        i = test_per * len(self.files) // 100
        self.trainset= self.files[i:]
        self.testset = self.files[:i]

        return self


class Transform(object):
    def __init__(self):
        pass

    def grayscale(self, img):
        try:
            new = img.copy().convert("L")
            return new
        except Exception as e:
            print(e)
            raise e

    def resize(self, img, size):
        try:
            new = img.copy().resize(size)
            return new
        except Exception as e:
            print(e)
            raise e

def set_img_path(path):
    global IMG_PATH
    IMG_PATH = path

def set_img_res(res):
    global IMG_RES
    IMG_RES = res
