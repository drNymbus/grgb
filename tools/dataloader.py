import sys
import os
import numpy
import random
from PIL import Image

from torch.utils.data import Dataset

IMG_RES = [48,48]
IMG_PATH = "data/paysage_48x48/"

class Dataloader(object):
    def __init__(self, path, test_per=0, mode=None):#, gray=False):
        self.path = path
        self.files = []
        for _,_,files in os.walk(self.path):
            for file in files:
                self.files.append(file)

        self.transform = Transform()
        # self.gray = gray # determine if pictures are in grayscale or not
        self.mode = mode

        random.shuffle(self.files)
        i = test_per * len(self.files) // 100
        self.trainset= self.files[i:]
        self.testset = self.files[:i]

    def set_mode(self, mode):
        if (mode == "train" or mode == "test"):
            self.mode = mode

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

    # def get_len_files(self):
    #     ''' return the total length of files '''
    #     return len(self.files)

    # def get_filename(self, i):
    #     ''' return filename self.files at index i '''
    #     return self.files[i]

    def __getitem__(self, i):
        if self.mode == "train":
            img = self.load_image(self.path + self.trainset[i])
            gray = self.transform.grayscale(img)

            item = {}
            item["path"] = self.path + self.trainset[i]
            item["data"] = numpy.asarray(gray)
            item["label"] = numpy.asarray(img)
            return item

        elif self.mode == "test":
            img = self.load_image(self.path + self.testset[i])
            gray = self.transform.grayscale(img)

            item = {}
            item["path"] = self.path + self.testset[i]
            item["data"] = numpy.asarray(gray)
            item["label"] = numpy.asarray(img)
            return item

        else:
            return self.files[i]

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
