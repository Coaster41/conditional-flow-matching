# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import sys
sys.path.append('../../..')



import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from torchcfm.models.unet.unet import UNetModelWrapper
from utils_cifar import generate_class_images, generate_mnist_class_images, generate_mnist_class_samples

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("model_path", "./results/fm_p020fm/fm_cifar10_weights_step_400000.pt", help="output_directory")
flags.DEFINE_string("model_name", "p020fm", help="flow matching model type")
flags.DEFINE_integer("num_gen", 100, help="number of samples to generate")
flags.DEFINE_string("save_dir", "./gen_images/", help="directory to save images")
flags.DEFINE_integer("num_classes", 10, help="number of classes")
flags.DEFINE_integer("batch_size", 1024, help="number of images to generate at once")
flags.DEFINE_bool("mnist", False, help="use flag to toggle on mnist")
FLAGS(sys.argv)


# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if FLAGS.mnist:
    new_net = UNetModelWrapper(
            dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True
        ).to(device)
else:
    new_net = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
        num_classes=FLAGS.num_classes,
        class_cond=True
    ).to(device)


# Load the model
# PATH = f"{FLAGS.input_dir}/{FLAGS.model}/{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt"
PATH = FLAGS.model_path
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["net_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()
for class_to_gen in range(FLAGS.num_classes):
    images_gen = 0
    while images_gen < FLAGS.num_gen:
        num_images = min(FLAGS.num_gen-images_gen, FLAGS.batch_size)
        if FLAGS.mnist:
            generate_mnist_class_images(new_net, FLAGS.save_dir, FLAGS.model_name, class_to_gen, num_images, images_gen)
        else:
            generate_class_images(new_net, FLAGS.save_dir, FLAGS.model_name, class_to_gen, num_images, images_gen)
        images_gen += num_images



# if FLAGS.mnist:
#     generate_mnist_class_samples(new_net, False, FLAGS.save_dir, "","")
