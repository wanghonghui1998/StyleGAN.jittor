import argparse
import numpy as np
from PIL import Image
from numpy.core.numerictypes import sctype2char

# import torch
import jittor as jt

from generate_grid import adjust_dynamic_range
from models.GAN import Generator

jt.flags.use_cuda = 1
def draw_interp_figure(png, gen, out_depth, src_seeds, dst_seeds, psis):
    w = h = 2 ** (out_depth + 2)
    latent_size = gen.g_mapping.latent_size

    # with torch.no_grad():
    with jt.no_grad():
        src_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size) for seed in src_seeds])
        dst_latents_np = np.stack([np.random.RandomState(seed).randn(latent_size) for seed in dst_seeds])
        # latents = torch.from_numpy(latents_np.astype(np.float32))
        src_latents = jt.array(src_latents_np.astype(np.float32))
        dst_latents = jt.array(dst_latents_np.astype(np.float32))
        # dlatents = gen.g_mapping(latents).detach().numpy()  # [seed, layer, component]
        src_dlatents = gen.g_mapping(src_latents).detach()  # [seed, layer, component]
        dst_dlatents = gen.g_mapping(dst_latents).detach()
        # dlatent_avg = gen.truncation.avg_latent.numpy()  # [component]
        # dlatent_avg = gen.truncation.avg_latent.data  # [component]

        canvas = Image.new('RGB', (w * len(psis), h * len(src_seeds)), 'white')
        # for row, dlatent in enumerate(list(dlatents)):
        for col, alpha in enumerate(psis):
            # row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
            col_dlatents = alpha * src_dlatents + (1-alpha) * dst_dlatents
            # row_dlatents = torch.from_numpy(row_dlatents.astype(np.float32))
            # row_dlatents = jt.array(row_dlatents.astype(np.float32))
            col_images = gen.g_synthesis(col_dlatents, depth=out_depth, alpha=1)
            for row, image in enumerate(list(col_images)):
                image = adjust_dynamic_range(image)
                # image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                image = image.multiply(255).clamp(0, 255).permute(1, 2, 0).data.astype(np.uint8)
                canvas.paste(Image.fromarray(image, 'RGB'), (col * w, row * h))
        canvas.save(png)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.model.gen.use_noise=False
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
    # seeds for face
    # src_seeds=[11,37, 44, 86]
    # dst_seeds=[21,36,1515,16]
    # seed for color
    src_seeds=[113,327, 414, 16]
    dst_seeds=[221,326,1351,149]
    draw_interp_figure(args.output, gen, out_depth=5,
                                 src_seeds=src_seeds, dst_seeds=dst_seeds, psis=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9,1])
    
    print('Done.')


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/sample.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)
    parser.add_argument("--output", action="store", type=str,
                        default="./output/color128-style-mixing.png",
                        help="path to the output path for the frames")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_arguments())
