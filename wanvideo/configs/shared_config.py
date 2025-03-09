# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5 model configuration
wan_shared_cfg.t5_model = 'umt5_xxl'  # Specifies the T5 model variant to be used
wan_shared_cfg.t5_dtype = torch.bfloat16  # Data type for T5 model weights and computations
wan_shared_cfg.text_len = 512  # Maximum length of input text tokens

# transformer configuration
wan_shared_cfg.param_dtype = torch.bfloat16  # Data type for transformer model parameters

# inference configuration
wan_shared_cfg.num_train_timesteps = 1000  # Number of timesteps for training diffusion models
wan_shared_cfg.sample_fps = 16  # Frames per second for video sampling
wan_shared_cfg.sample_neg_prompt = (
    '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
)  # Negative prompt for guiding the generation process