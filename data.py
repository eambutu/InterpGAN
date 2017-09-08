from torch.utils.data import DataLoader
from PIL import Image
import misc
import numpy as np
import torch
import torchvision.transforms as transforms
import os


# Constants
SAMPLE_DIR = './anime_sample/'
TRANSFORM = transforms.Compose([
    misc.Scale((256,144)),
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (1,1,1))])
SPLITS_DIR = './splits/'


# Check if directory exists
if not os.path.exists(SPLITS_DIR):
    os.makedirs(SPLITS_DIR)


# Helper functions
def get_split_path(prefix, vsn='0'):
    split_path = SPLITS_DIR + prefix + '_' + vsn + '.txt'
    return split_path

def save_split(vsn='0'):
    split_train = get_split_path('train', vsn)
    split_test = get_split_path('test', vsn)
    if not (os.path.exists(split_train) and os.path.exists(split_test)):
        num_files = len([name for name in os.listdir(SAMPLE_DIR)])
        num_itv = num_files // 3
        split_idx = int(num_itv * 0.8)
        itv_list = np.random.permutation(num_itv)
        train_itvs, test_itvs = itv_list[0:split_idx], itv_list[split_idx:]
        with open(split_train, 'w') as fin:
            for itv in train_itvs:
                fin.write(str(3 * itv + 1) + '\n')
        with open(split_test, 'w') as fin:
            for itv in test_itvs:
                fin.write(str(3 * itv + 1) + '\n')
        print('Data has been split into:\n  %s\n  %s' % (split_train, split_test))

def load_img(idx):
    sample = SAMPLE_DIR + 'anime_sample_' + str(idx + 1).zfill(4) + '.jpg'
    with open(sample, 'rb') as fin:
        with Image.open(fin) as img:
            return img.convert('RGB')

def init_data_loader(vsn, batch_size, test=False, use_gpu=False):
    split_file = get_split_path('test' if test else 'train', vsn)
    processed_data = GetDataSamples(split_file, use_gpu=use_gpu)
    return DataLoader(processed_data, batch_size)


# Generate data
class GetDataSamples:

    def __init__(self, split_file, use_gpu=False):
        with open(split_file, 'r') as fin:
            self.data_lines = [int(line) for line in fin.readlines()]
        self.num_samples = len(self.data_lines)
        self.use_gpu = use_gpu

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        center_idx = self.data_lines[idx]
        left_idx = center_idx - 1
        right_idx = center_idx + 1
        left = TRANSFORM(load_img(left_idx))
        right = TRANSFORM(load_img(right_idx))
        center = TRANSFORM(load_img(center_idx))
        if self.use_gpu:
            left, right, center = left.cuda(), right.cuda(), center.cuda()
        left_right = torch.cat((left, right), 0)
        sample = {'start_end_frames': left_right, 'middle_frame': center}
        return sample
