
import os
import argparse
import numpy as np

# import torch
# from torchvision.utils import save_image
import jittor as jt

from models.GAN import Generator

jt.flags.use_cuda = 1
def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/sample.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)
    parser.add_argument("--n_row", action="store", type=int,
                        default=4, help="number of synchronized grids to be generated")
    parser.add_argument("--n_col", action="store", type=int,
                        default=4, help="number of synchronized grids to be generated")
    parser.add_argument("--output_dir", action="store", type=str,
                        default="./",
                        help="path to the output directory for the frames")

    args = parser.parse_args()

    return args


def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * jt.array(scale) + jt.array(bias)
    # return torch.clamp(data, min=0, max=1)
    return jt.clamp(data, min_v=0, max_v=1)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    print("Creating generator object ...")
    # create the generator object
    gen = Generator(resolution=opt.dataset.resolution,
                    num_channels=opt.dataset.channels,
                    structure=opt.structure,
                    **opt.model.gen)

    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    # gen.load_state_dict(torch.load(args.generator_file))
    gen.load(args.generator_file)

    # path for saving the files:
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    latent_size = opt.model.gen.latent_size
    out_depth = int(np.log2(opt.dataset.resolution)) - 2

    print("Generating scale synchronized images ...")
    # generate the images:
    # with torch.no_grad():
    with jt.no_grad():
        # point = torch.randn(args.n_row * args.n_col, latent_size)
        np.random.seed(1000)
        point = np.random.randn(args.n_row * args.n_col, latent_size)
        # point = (point / point.norm()) * (latent_size ** 0.5)
        point = (point / np.linalg.norm(point)) * (latent_size ** 0.5)
        point = jt.array(point, dtype='float32')
        ss_image = gen(point, depth=out_depth, alpha=1)
        # color adjust the generated image:
        ss_image = adjust_dynamic_range(ss_image)
    print("gen done")
    # save the ss_image in the directory
    # ss_image = torch.from_numpy(ss_image.data)
    # save_image(ss_image, os.path.join(save_path, "grid.png"), nrow=args.n_row,
    #             normalize=True, scale_each=True, pad_value=128, padding=1)
    jt.save_image_my(ss_image, os.path.join(save_path, "grid.png"), nrow=args.n_row, normalize=True, scale_each=True, pad_value=128, padding=1)

    print('Done.')


if __name__ == '__main__':
    main(parse_arguments())
