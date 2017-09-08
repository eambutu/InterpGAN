from torch.autograd import Variable
import data
import os
import torch
import torch.optim as optim
import torchvision.utils as vutils


# Constants
DLR = 1e-3
GLR = 7e-7
DMOMENTUM = 0.9
BETAS = (0.499, 0.999)
LOGS_DIR = './logs/'
RESULTS_DIR = './results/'


# Check if directory exists
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# GAN code
class GAN:

    def __init__(self, disc, gen, use_gpu=False):
        self.disc = disc
        self.gen = gen
        # TODO: test between SGD and Adam for discriminator
        self.d_optimizer = optim.SGD(disc.parameters(), lr=DLR, momentum=DMOMENTUM)
        # self.d_optimizer = optim.Adam(disc.parameters(), lr=DLR, betas=BETAS)
        self.g_optimizer = optim.Adam(gen.parameters(), lr=GLR, betas=BETAS)
        self.use_gpu = use_gpu

    def sanity(self, vsn, ep, num_eps, i, num_batches,
               disc_loss, gen_loss, real_mean, fake_mean, gen_mean):
        msg = '[%d/%d][%d/%d] DL: %f GL: %f D(x): %.4f D(G(z)): %.4f ~ %.4f\n' % (
            ep+1, num_eps, i+1, num_batches,
            disc_loss, gen_loss, real_mean, fake_mean, gen_mean)
        print(msg),
        with open(LOGS_DIR + 'log_' + vsn, 'a') as fin:
            fin.write(msg)
            fin.close()
        self.disc.check_weights()
        self.gen.check_weights()

    def disc_train_step(self, batch):
        real_triple = torch.cat(
            [batch['start_end_frames'], batch['middle_frame']], 1)
        real_preds = self.disc(Variable(real_triple))
        fake_triple = self.gen.get_samples(batch, use_gpu=self.use_gpu)
        fake_triple.data = torch.cat(
            [batch['start_end_frames'], fake_triple.data[:,3:6,:,:]], 1)
        fake_preds = self.disc(fake_triple)
        disc_err_example, disc_err_gen = self.disc.cost(
            real_preds, fake_preds, use_gpu=self.use_gpu)
        self.d_optimizer.zero_grad()
        disc_err_example.backward()
        disc_err_gen.backward()
        self.d_optimizer.step()
        disc_err = disc_err_example + disc_err_gen
        return disc_err.data[0], real_preds.data.mean(), fake_preds.data.mean()

    def gen_train_step(self, batch):
        gen_triple = self.gen.get_samples(batch, use_gpu=self.use_gpu)
        gen_triple.data = torch.cat(
            [batch['start_end_frames'], gen_triple.data[:,3:6,:,:]], 1)
        gen_preds = self.disc(gen_triple)
        gen_err = self.gen.cost(gen_preds, use_gpu=self.use_gpu)
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        gen_err.backward()
        self.g_optimizer.step()
        return gen_err.data[0], gen_preds.data.mean()

    def train(self, vsn, num_eps=1000, batch_size=32, num_gen=2, num_disc=1):
        loaded_data = data.init_data_loader(vsn, batch_size, use_gpu=self.use_gpu)
        for ep in range(num_eps):
            for i, batch in enumerate(loaded_data):
                gen_idx, disc_loss, gen_loss = 0, 0, 0
                real_mean, fake_mean, gen_mean = 0, 0, 0

                # Disc training loop
                for _ in range(num_disc):
                    disc_loss, real_mean, fake_mean = self.disc_train_step(batch)

                # Gen training loop
                while gen_idx < num_gen or (gen_idx < 3 and real_mean > gen_mean):
                    gen_loss, gen_mean = self.gen_train_step(batch)
                    gen_idx += 1

                # Logging
                self.sanity(vsn, ep, num_eps, i, len(loaded_data),
                            disc_loss, gen_loss, real_mean, fake_mean, gen_mean)
            self.disc.save_weights(vsn, str(ep))
            self.gen.save_weights(vsn, str(ep))

    def test(self, vsn, epoch, batch_size=32):
        loaded_data = data.init_data_loader(vsn, batch_size, test=True, use_gpu=self.use_gpu)
        for i, batch in enumerate(loaded_data):
            start_end_frames = Variable(batch['start_end_frames'])
            start_frame = start_end_frames[:, 0:3, :, :]
            end_frame = start_end_frames[:, 3:6, :, :]
            gen_frame = self.gen(True, start_end_frames)[:, 3:6, :, :]
            frames = (torch.cat([start_frame, gen_frame, end_frame]) + 1) / 2
            vutils.save_image(frames.data, RESULTS_DIR + vsn + '_' + epoch + '_%d.png' % i)
