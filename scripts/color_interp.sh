cd ..

CUDA_VISIBLE_DEVICES=2, \
python generate_interp_figure.py \
--config /home/whh/Experiments/StyleGAN/color_symbol_7k/sample_color_128_affine_false/sample_color_128_affine_false.yaml \
--generator_file /home/whh/Experiments/StyleGAN/color_symbol_7k/sample_color_128_affine_false/models/GAN_GEN_SHADOW_5_240.pkl \
--output ./output/color-128-false-interp.png

# CUDA_VISIBLE_DEVICES=2, \
# python generate_mixing_figure.py \
# --config /home/whh/Experiments/StyleGAN/color_symbol_7k/sample_color_128/sample_color_128.yaml \
# --generator_file /home/whh/Experiments/StyleGAN/color_symbol_7k/sample_color_128/models/GAN_GEN_SHADOW_5_240.pkl \
# --output ./output/color-128-style-mixing.png