import numpy
from PIL import Image

IMG_RES = [153, 102]

class Transform():
    def __init__(self, path):
        self.path = path
        self.img = None
        self.rimg = None
        self.gray = None
        self.rgray = None
        self.size = {"original" : None, "resize" : None}

    def to_gray(self):
        self.gray = self.img.copy().convert("L")
        self.size["original"] = self.gray.size
        return self

    def set_size(self, size):
        global IMG_RES
        IMG_RES[0], IMG_RES[1] = size[0], size[1]
        return self

    def resize(self, w, h):
        self.rimg = self.img.copy().resize((w,h), Image.ANTIALIAS)
        self.rgray = self.gray.copy().resize((w,h), Image.ANTIALIAS)
        self.size["resize"] = self.rgray.size
        return self

    def set_path(self, path):
        self.path = path
        return self

    def load_path(self, filepath):
        self.img = Image.open(self.path + filepath)
        if (self.img.mode == 'L'):
            self.gray = self.img
            self.img = self.img.copy().convert("RGB")
        else:
            self.to_gray()
        iw, ih = self.img.size
        if iw > ih:
            self.resize(IMG_RES[0], IMG_RES[1])
        else:
            self.resize(IMG_RES[1], IMG_RES[0])
        return self

    def get_item(self, path):
        self.load_path(path)
        item = {
            "data" : numpy.asarray(self.rgray),
            "label" : numpy.asarray(self.rimg),
            "path" : self.path + path
        }
        return item

    def save_item(self, path, size):
        self.resize(size[0], size[1])
        self.rimg.save(path)
        return self
