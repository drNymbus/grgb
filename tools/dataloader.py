import sys
import os
import numpy
from PIL import Image

from torch.utils.data import Dataset

from tools.transform import Transform

IMG_RES = [153, 102]
IMG_PATH = "data/pictures"
# JSON_PATH = "data/json/urls"

class DataLoader(Dataset):
    def __init__(self, path, transform=False, dim=None):
        self.root = path
        self.size = IMG_RES
        if (dim is not None):
            # self.root = path + (str(dim[0]) + 'x' + str(dim[1])) + '/'
            self.size = dim
        # print(self.root, self.size)
        self.dirs = []
        self.files = {}

        self.transform = None
        if transform:
            self.transform = Transform(self.root)
            self.transform.set_size(self.size)

        self.pathname = None
        self.item = {}
        self.index = {"dir":0, "file":0}

        dirs = os.listdir(self.root)
        for dir in dirs:
            self.dirs.append(dir)
            self.files[dir] = []

            for _,_,files in os.walk(self.root + dir):
                for file in files:
                    self.files[dir].append(file)

    def __len__(self):
        count = 0
        for key in self.files:
            count += len(self.files[key])
        return count

    def load_image(self, path): # path ==> root/path
        self.pathname = self.root + path
        if self.transform is None:
            img = Image.open(self.pathname)
            gray = img.copy().convert("L")

            self.item["data"] = numpy.asarray(img)
            self.item["label"] = numpy.asarray(gray)
            self.item["path"] = self.pathname
            self.item["index"] = self.index
        else:
            self.item = self.transform.get_item(path, self.index)
        return self

    def free_image(self):
        del self.item["data"]
        del self.item["label"]
        del self.item["path"]
        del self.item["index"]
        self.pathname = None
        return self

    def get_image(self, path):
        pathname = self.root + path
        if pathname != self.pathname:
            if self.pathname is not None:
                self.free_image()
            self.load_image(path)
        return self.item

    def get_img_idx(self):
        dir = self.dirs[self.index["dir"]]
        file = self.files[dir][self.index["file"]]
        path = dir + '/' + file
        return self.get_image(path)

    def set_index(self, i):
        count = 0
        for i_dir in range(len(self.dirs)):
            dir = self.dirs[i_dir]
            for i_file in range(len(self.files[dir])):
                if count == i:
                    self.index["dir"] = i_dir
                    self.index["file"] = i_file
                    return True
                count += 1
        return False

    def __getitem__(self, idx):
        if self.set_index(idx):
            item = self.get_img_idx()
            return item
        return None

    def get_img_size(self):
        return self.size

    def save_resize(self, w, h, out=None):
        if (self.transform is not None):
            path = self.root + '_' + str(w) + 'x' + str(h) + '/'
            try:
                os.mkdir(path)
                for dir in self.dirs:
                    os.mkdir(path + dir)
            except:
                pass
            i = len(self)-1
            while (i >= 0):
                self.set_index(i)
                self.get_img_idx()
                dirname = self.dirs[self.index["dir"]] + '/'
                filename = self.pathname.split('/')[::-1][0]
                print(i, path + dirname + filename)
                self.transform.save_item(path + dirname + filename, [w,h])
                i -= 1
        return self

if __name__ == "__main__":
    # size = [int(sys.argv[1]), int(sys.argv[2])]
    # data = DataLoader(IMG_PATH, transform=True)
    # data.save_resize(size[0], size[1])
    # s = data.get_img_size()
    # print(s)
    pass
