cd ..

CUDA_VISIBLE_DEVICES=2, \
python generate_interp_figure.py \
--config /home/whh/Experiments/StyleGAN/FFHQ128/sample_ffhq_128_affine_false/sample_ffhq_128_affine_fales.yaml \
--generator_file /home/whh/Experiments/StyleGAN/FFHQ128/sample_ffhq_128_affine_false/models/GAN_GEN_SHADOW_5_64.pkl \
--output ./output/face-128-false-interp.png

# CUDA_VISIBLE_DEVICES=2, \
# python generate_mixing_figure.py \
# --config /home/whh/Experiments/StyleGAN/color_symbol_7k/sample_color_128/sample_color_128.yaml \
# --generator_file /home/whh/Experiments/StyleGAN/color_symbol_7k/sample_color_128/models/GAN_GEN_SHADOW_5_240.pkl \
# --output ./output/color-128-style-mixing.png