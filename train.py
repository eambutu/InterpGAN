import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from data import FramesDataset
from model import Generator, Discriminator


random.seed(1337)

save_folder = './saves'
d_steps = 2
g_steps = 1
d_learning_rate = 2e-5
g_learning_rate = 2e-5
z_dim = 256
d_momentum = 0.9
betas = (0.9, 0.999)
num_epochs = 75
batch_size = 4
label_flip_prob = 0.05

def get_distribution_sampler(mu, sigma):
    return lambda m, n: torch.Tensor(np.random.normal(mu, sigma, (m, n)))  # Gaussian

def extract(v):
    return v.data.storage().tolist()

def smooth_false_label():
    return random.uniform(0, 0.3)

def smooth_true_label():
    return random.uniform(0.7, 1)

def main():
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--gpu', help='Use gpu for training', default=False,
                        action='store_true')
    args = parser.parse_args()

    data = FramesDataset(3372, './anime_sample_train.txt', './anime_sample')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    G = Generator()
    D = Discriminator()
    criterion = nn.BCELoss()
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=d_momentum)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=betas)
    g_sampler = get_distribution_sampler(0, 1) 

    if args.gpu:
        G.cuda()
        D.cuda()
        criterion.cuda()

    for epoch in range(num_epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            smooth_truth = torch.Tensor([smooth_true_label() for i in range(batch_size)])
            smooth_false = torch.Tensor([smooth_false_label() for i in range(batch_size)])
            if args.gpu:
                sample_batched['start_end_frames'] = sample_batched['start_end_frames'].cuda()
                sample_batched['middle_frame'] = sample_batched['middle_frame'].cuda()
                smooth_truth = smooth_truth.cuda()
                smooth_false = smooth_false.cuda()
            start_end_frames = Variable(sample_batched['start_end_frames']).float()

            # Whether or not to flip label for discriminator
            flip_label = (random.random() < label_flip_prob)

            for d_index in range(d_steps):
                D.zero_grad()

                # Train D on real
                real_data = torch.cat((sample_batched['start_end_frames'],
                                       sample_batched['middle_frame']), 1)
                d_real_data = Variable(real_data)
                d_real_data = d_real_data.float()
                d_real_decision = D(d_real_data)

                # Train D on fake
                d_gen_input = Variable(g_sampler(4, z_dim))
                d_fake_frame = G(d_gen_input, start_end_frames).detach()
                d_fake_frames = Variable(torch.cat((sample_batched['start_end_frames'],
                                                    d_fake_frame.data.double()), 1))
                d_fake_frames = d_fake_frames.float()
                d_fake_decision = D(d_fake_frames)

                if not flip_label:
                    d_real_error = criterion(d_real_decision, Variable(smooth_truth))
                    d_fake_error = criterion(d_fake_decision, Variable(smooth_false))
                else:
                    d_real_error = criterion(d_real_decision, Variable(smooth_false))
                    d_fake_error = criterion(d_fake_decision, Variable(smooth_truth))
                d_real_error.backward()
                d_fake_error.backward()

                d_error = d_real_error + d_fake_error
                d_optimizer.step()

            for g_index in range(g_steps):
                G.zero_grad()

                gen_input = Variable(g_sampler(4, z_dim))
                g_fake_frame = G(gen_input, start_end_frames)
                g_fake_frames = Variable(torch.cat((sample_batched['start_end_frames'],
                                                    g_fake_frame.data.double()), 1))
                g_fake_frames = g_fake_frames.float()
                dg_fake_decision = D(g_fake_frames)
                g_error = criterion(dg_fake_decision, Variable(smooth_truth))

                g_error.backward()
                g_optimizer.step()

            if i_batch % 10 == 0:
                print(dg_fake_decision)
            if i_batch % 100 == 0:
                # Print loss
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, num_epochs, i_batch, len(dataloader),
                         d_error.data[0], g_error.data[0]))
                # Save some images
                # vutils.save_image(sample_batched['middle_frame'][0],
                #                   '%s/real_samples.png' % save_folder,
                #                   normalize=False)
                # vutils.save_image(g_fake_frame[0],
                #                   '%s/fake_samples_epoch_%03d.png' % (save_folder, epoch),
                #                   normalize=False)
        
        torch.save(G.state_dict(), '%s/gen_epoch_%d.pth' % (save_folder, epoch))
        torch.save(D.state_dict(), '%s/dis_epoch_%d.pth' % (save_folder, epoch))
                      

if __name__=='__main__':
    main()
