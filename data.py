from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np


class FramesDataset(Dataset):
    def __init__(self, num_images, dataset_file, root_dir, transform=None):
        self.num_images = num_images
        self.root_dir = root_dir

        # Resize the image to nicer looking dimensions, still 16 x 9
        self.transform = lambda x: self.__transform__(x) 

        with open(dataset_file, 'r') as fin:
            self.data_lines = fin.readlines()
            self.data_lines = [line.strip() for line in self.data_lines]

    def __len__(self):
        return self.num_images

    def __transform__(self, image):
        # Resize the image to smaller dimension, still 16 x 9
        temp = resize(image, (112, 256, 3))
        # Normalize between -1 and 1
        temp = temp / 127.5
        temp = temp - 1
        return temp

    def __getitem__(self, idx):
        # first and second imgs are first and third frames. third imgs is the in
        # between frame
        imgs_path = self.data_lines[idx].split(' ')
        imgs = [io.imread(self.root_dir + '/' + img) for img in imgs_path]

        imgs = [self.transform(img) for img in imgs]

        # RGB image stored as MxNx3, convert to 3xMxN
        imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]

        sample = {'start_end_frames': np.concatenate((imgs[0], imgs[1]), axis=0),
                  'middle_frame': imgs[2]}

        return sample
