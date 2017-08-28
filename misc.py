import collections
import numpy as np
from PIL import Image, ImageOps

class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

class AddGaussianNoise(object):
    """Add gaussian noise to a numpy.ndarray (H x W x C)
    """

    def __init__(self, mean, sigma, random_state=np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.mean, collections.Sequence, random_state=self.random_state):
            mean = random_num_generator(self.mean)
        else:
            mean = self.mean
        row, col, ch = image.shape
        gauss = self.random_state.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        image += gauss
        return image
