import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from data import FramesDataset
from model import Generator, Discriminator

import sys

sys.path.append('/home/phillip/Projects/vision/torchvision')

import utils

weights_file_path = './saves/gen_epoch_24.pth'
test_write_path = './samples/'

def main():
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--gpu', help='Use gpu for training', default=False,
                        action='store_true')
    args = parser.parse_args()

    data = FramesDataset(100, './anime_sample_test.txt', './anime_sample')
    dataloader = DataLoader(data, batch_size=1)

    G = Generator()
    if args.gpu:
        G_weights = torch.load(weights_file_path)
    else:
        G_weights = torch.load(weights_file_path, map_location=lambda storage,
                               loc: storage)
    G.load_state_dict(G_weights)

    for i_batch, sample_batched in enumerate(dataloader):
        start_end_frames = Variable(sample_batched['start_end_frames']).float()

        # Dummy variable for now
        gen_frame = G(True, start_end_frames)
        start_frame = start_end_frames[:, 0:3, :, :]
        end_frame = start_end_frames[:, 3:6, :, :]
        frames = torch.cat([start_frame, gen_frame, end_frame])
        frames = (frames + 1) / 2
        frames = frames.data
        utils.save_image(frames, test_write_path + '%d.png' % (i_batch))

if __name__ == '__main__':
    main()
