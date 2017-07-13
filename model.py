import torch.nn as nn
import torch.nn.functional as F

# Taken from DCGAN pytorch tutorial
def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def fc(input_dim, output_dim, bn=True):
    layers = []
    layers.append(nn.Linear(input_dim, output_dim))
    if bn:
        layers.append(nn.BatchNorm1d(output_dim))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    """Generator containing 7 deconvolutional layers."""
    def __init__(self, z_dim=256, image_w=256, image_h=112, conv_dim=64):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(6, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, z, frame_concat, image_w=256, image_h=112):
        # let's ignore the z for now
        out = F.leaky_relu(self.conv1(frame_concat), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 128, 32, 14)
        out = F.leaky_relu(self.deconv2(out), 0.05)  # (?, 64, 64, 28)
        out = F.leaky_relu(self.deconv3(out), 0.05)  # (?, 32, 128, 56)
        out = F.tanh(self.deconv4(out))             # (?, 3, 256, 112)
        return out

class Discriminator(nn.Module):
    """Discriminator containing 4 convolutional layers."""
    def __init__(self, image_w=256, image_h=112, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(9, conv_dim, 5, pad=2, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 5, pad=2)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 3)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 3)
        self.conv5 = conv(conv_dim*8, conv_dim*2, 3, stride=1)
        self.fc = fc(conv_dim * 2 * (image_w // 16) * (image_h // 16), 1)

    def forward(self, x, image_w=256, image_h=112):                         # For image shape 256 x 112
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 32, 128, 56)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 64, 64, 28)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 128, 32, 14)
        out = F.leaky_relu(self.conv4(out), 0.05)  # (?, 256, 16, 7)
        out = F.leaky_relu(self.conv5(out), 0.05)
        out = out.view(-1, self.conv_dim * 2 * (image_w // 16) * (image_h // 16)) # (?, 28672)
        out = F.sigmoid(self.fc(out))
        return out
