import argparse
import data
import gan
import model


# Constants
VSN = '7'
BATCH_SIZE = 64
NUM_EPS = 1000
NUM_GEN = 2
NUM_DISC = 1


# Helper functions
def get_args():
    parser = argparse.ArgumentParser(description='Help animators animate anime.')
    parser.add_argument('--gpu', help='Use GPU for training',
                        default=False, action='store_true')
    parser.add_argument('--test', help='Test GAN instead of train',
                        default=False, action='store_true')
    parser.add_argument('--load', help='Pass version and epoch for weight loading',
                        type=str, nargs=2)
    args = parser.parse_args()
    return args


# Main loop
if __name__ == '__main__':
    # Get arguments
    args = get_args()

    # Set up models
    disc = model.Discriminator()
    gen = model.Generator()
    if args.load is not None:
        disc.load_weights(args.load[0], args.load[1], use_gpu=args.gpu)
        gen.load_weights(args.load[0], args.load[1], use_gpu=args.gpu)
    if args.gpu:
        disc.cuda()
        gen.cuda()

    # Run GAN
    GAN = gan.GAN(disc, gen, use_gpu=args.gpu)
    if args.test:
        if args.load is None:
            print('Test requires passing load!')
        else:
            GAN.test(args.load[0], args.load[1], batch_size=BATCH_SIZE)
    else:
        data.save_split(vsn=VSN)
        GAN.train(VSN, num_eps=NUM_EPS, batch_size=BATCH_SIZE,
                  num_gen=NUM_GEN, num_disc=NUM_DISC)
