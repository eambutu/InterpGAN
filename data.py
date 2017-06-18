from torch.utils.data import Dataset
from skimage import io
import numpy as np


class FramesDataset(Dataset):
    def __init__(self, num_images, dataset_file, root_dir, transform=None):
        self.num_images = num_images
        self.root_dir = root_dir
        self.transform = transform

        with open(dataset_file, 'r') as fin:
            self.data_lines = fin.readlines()

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # first and second imgs are first and third frames. third imgs is the in
        # between frame
        imgs_path = self.data_lines[idx].split(' ')
        imgs = [io.imread(self.root_dir + '/' + img) for img in imgs_path]

        # RGB image stored as MxNx3
        imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]

        sample = {'start_end_frames': np.concatenate((imgs[0], imgs[1]), axis=0),
                  'middle_frame': imgs[2]}

        return sample
