from torch.autograd.variable import Variable
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# Constants
WEIGHTS_DIR = './weights/'


# Check if directory exists
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)


# Helper functions
def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, dropout_p=0):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    # TODO: figure out why BatchNorm2d makes disc converge to half.
    # if bn:
    #     layers.append(nn.BatchNorm2d(c_out))
    if dropout_p > 0:
        layers.append(nn.Dropout2d(dropout_p, True))
    return nn.Sequential(*layers)

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def fc(input_dim, output_dim, bn=True, dropout_p=0):
    layers = []
    layers.append(nn.Linear(input_dim, output_dim, bias=False))
    # TODO: figure out why BatchNorm2d makes disc converge to half.
    # if bn:
    #     layers.append(nn.BatchNorm1d(output_dim))
    if dropout_p > 0:
        layers.append(nn.Dropout2d(dropout_p, True))
    return nn.Sequential(*layers)

def criterion(outputs, labels, use_gpu=False):
    loss_fun = torch.nn.BCELoss().cuda() if use_gpu else torch.nn.BCELoss()
    loss = loss_fun(outputs, labels)
    return loss

def get_labels(size, low, high, use_gpu=False):
    labels = torch.FloatTensor(size)
    labels = labels.fill_(low) if low == high else labels.uniform_(low, high)
    labels = Variable(labels.cuda() if use_gpu else labels)
    return labels

def get_distribution_sampler(mu, sigma, use_gpu=False):
    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    sampler = lambda m, n: FloatTensor(np.random.normal(mu, sigma, (m, n)))
    return sampler

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_weights_path(prefix, vsn, epoch):
    weights_path = WEIGHTS_DIR + prefix + '_' + vsn + '_' + epoch + '.pth'
    return weights_path


# Parent neural net module
class NeuralNet(nn.Module):

    def load_weights(self, vsn, epoch, use_gpu=False):
        weights_file = get_weights_path(self.prefix, vsn, epoch)
        if use_gpu:
            weights = torch.load(weights_file)
        else:
            weights = torch.load(
                weights_file, map_location=lambda storage, loc: storage)
        self.load_state_dict(weights)
        print('Model loaded:\n  %s' % weights_file)

    def save_weights(self, vsn, epoch):
        weights_file = get_weights_path(self.prefix, vsn, epoch)
        torch.save(self.state_dict(), weights_file)
        print('Model weights saved:\n  %s' % weights_file)


# Generator from DCGAN PyTorch tutorial
class Generator(NeuralNet):

    def __init__(self, z_dim=256, conv_dim=64):
        super(Generator, self).__init__()
        self.prefix = 'gen'
        self.z_dim = z_dim
        self.conv_dim = conv_dim

        # For image shape 256 x 144
        self.conv1 = conv(6, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 9, 4, bn=False)
        self.apply(weights_init)

    def cost(self, fake_preds, use_gpu=False):
        labels_gen = get_labels(fake_preds.size(), 1, 1, use_gpu=use_gpu)
        cost_gen = criterion(fake_preds, labels_gen, use_gpu=use_gpu)
        return cost_gen

    def forward(self, z, frame_concat):
        out = F.leaky_relu(self.conv1(frame_concat), 0.02, inplace=True)
        out = F.leaky_relu(self.conv2(out), 0.02, inplace=True)
        out = F.leaky_relu(self.conv3(out), 0.02, inplace=True)
        out = F.leaky_relu(self.conv4(out), 0.02, inplace=True)
        out = F.leaky_relu(self.deconv1(out), 0.02, inplace=True)  # (?, 128, 32, 14)
        out = F.leaky_relu(self.deconv2(out), 0.02, inplace=True)  # (?, 64, 64, 28)
        out = F.leaky_relu(self.deconv3(out), 0.02, inplace=True)  # (?, 32, 128, 56)
        out = F.tanh(self.deconv4(out))                            # (?, 3, 256, 112)
        return out

    def get_samples(self, batch, use_gpu=False):
        gen_sampler = get_distribution_sampler(0, 1, use_gpu)
        z_var = Variable(gen_sampler(4, self.z_dim))
        gen_input = Variable(batch['start_end_frames'])
        gen_samples = self(z_var, gen_input)
        return gen_samples

    def check_weights(self):
        print('  Generator...')
        print('    conv4-weigh: ' + str(self.conv4.__getitem__(0).weight.data.norm()))
        print('    conv4-grads: ' + str(self.conv4.__getitem__(0).weight.grad.data.norm()))
        print('    deconv4-weigh: ' + str(self.deconv4.__getitem__(0).weight.data.norm()))
        print('    deconv4-grads: ' + str(self.deconv4.__getitem__(0).weight.grad.data.norm()))


# Discriminator from DCGAN PyTorch tutorial
class Discriminator(NeuralNet):

    def __init__(self, image_w=256, image_h=144, conv_dim=64):
        super(Discriminator, self).__init__()
        self.prefix = 'disc'
        self.conv_dim = conv_dim

        # For image shape 256 x 144
        self.conv1 = conv(9, conv_dim, 5, pad=2, bn=False, dropout_p=0.1)
        self.conv2 = conv(conv_dim, conv_dim*2, 5, pad=2, dropout_p=0.25)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 3, dropout_p=0.25)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 3, dropout_p=0.5)
        self.conv5 = conv(conv_dim*8, conv_dim*2, 3, stride=1, dropout_p=0.5)
        self.fc = fc(conv_dim * 2 * (image_w // 16) * (image_h // 16), 1, dropout_p=0.5)
        self.apply(weights_init)

    def cost(self, outputs_example, outputs_gen, use_gpu=False):
        # TODO: not optimal because bad cache locality but okay for now.
        # We could search for something like torch.map(lambda x: 1-x, outputs_gen).
        labels_gen = get_labels(outputs_gen.size(), -0.2, 0.2, use_gpu=use_gpu)
        cost_gen = criterion(outputs_gen, labels_gen, use_gpu=use_gpu)
        labels_example = get_labels(outputs_example.size(), 0.8, 1.2, use_gpu=use_gpu)
        cost_example = criterion(outputs_example, labels_example, use_gpu=use_gpu)
        return cost_example, cost_gen

    def forward(self, x, image_w=256, image_h=144):
        out = F.leaky_relu(self.conv1(x), 0.02, inplace=True)    # (?, 32, 128, 56)
        out = F.leaky_relu(self.conv2(out), 0.02, inplace=True)  # (?, 64, 64, 28)
        out = F.leaky_relu(self.conv3(out), 0.02, inplace=True)  # (?, 128, 32, 14)
        out = F.leaky_relu(self.conv4(out), 0.02, inplace=True)  # (?, 256, 16, 7)
        out = self.conv5(out)
        out = out.view(-1, self.conv_dim * 2 * (image_w // 16) * (image_h // 16))
        out = F.sigmoid(self.fc(out))
        return out

    def check_weights(self):
        print('  Discriminator...')
        print('    conv5-weigh: ' + str(self.conv5.__getitem__(0).weight.data.norm()))
        print('    conv5-grads: ' + str(self.conv5.__getitem__(0).weight.grad.data.norm()))
        print('    fc-weigh: ' + str(self.fc.__getitem__(0).weight.data.norm()))
        print('    fc-grads: ' + str(self.fc.__getitem__(0).weight.grad.data.norm()))
