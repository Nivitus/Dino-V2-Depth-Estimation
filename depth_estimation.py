import numpy as np
import os, sys
import torch
import torchvision
import torch.nn.functional as F
import urllib
from PIL import Image
import matplotlib
import cv2
from torchvision import transforms

import mmcv
from mmcv.runner import load_checkpoint

import math
import itertools
from functools import partial
from dinov2.eval.depth.models import build_depther

# Utilities
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther

# Load pretrained backbone
BACKBONE_SIZE = "base" # in ("small", "base", "large" or "giant")

backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()

# Load pre trained depth model

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


HEAD_DATASET = "nyu" # in ("nyu", "kitti")
HEAD_TYPE = "dpt" # in ("linear", "linear4", "dpt")


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

cfg_str = load_config_from_url(head_config_url)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

model = create_depther(
    cfg,
    backbone_model=backbone_model,
    backbone_size=BACKBONE_SIZE,
    head_type=HEAD_TYPE,
)

load_checkpoint(model, head_checkpoint_url, map_location="cpu")
model.eval()
model.cuda()

# Load the sample Image
def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")
    
EXAMPLE_IMAGE_URL = "https://in.saint-gobain-glass.com/sites/in.saint-gobain-glass.com/files/inline-images/lacquered%20glss.jpg"

# image = load_image_from_url(EXAMPLE_IMAGE_URL)

window_width = 800
window_height = 500
custom_image_path = "image_path"
image = Image.open(custom_image_path)
image = image.resize((window_width, window_height)) 

# Convert to RGB if needed
image = image.convert("RGB")


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)

transform = make_depth_transform()

# For Image Inference
rescaled_image = image.resize((image.width, image.height))  # No rescaling needed for single image
transformed_image = transform(rescaled_image)
batch = transformed_image.unsqueeze(0).cuda()

with torch.inference_mode():
    result = model.whole_inference(batch, img_meta=None, rescale=True)
depth_image = render_depth(result.squeeze().cpu())

# Image inference mode
depth_image.show()

# Video inference mode

path = "video_path"
video_path = cv2.VideoCapture(path)

# Define the desired output window size
while True:
    ret, frame = video_path.read()
    if not ret:
        break

    transform = make_depth_transform()
    scale_factor = 0.4
    downsampled_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    transformed_frame = transform(downsampled_frame)
    batch = transformed_frame.unsqueeze(0).cuda()

    with torch.inference_mode():
        result = model.whole_inference(batch, img_meta=None, rescale=True)
    
    depth_image = render_depth(result.squeeze().cpu())
    resized_depth_image = cv2.resize(np.array(depth_image), (frame.shape[1], frame.shape[0]))

    # Resize the output window to the desired dimensions
    cv2.namedWindow('Depth Estimation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth Estimation', window_width, window_height)
    cv2.imshow('Depth Estimation', resized_depth_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_path.release()
cv2.destroyAllWindows()
