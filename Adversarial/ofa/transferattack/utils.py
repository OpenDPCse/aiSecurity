import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import pandas as pd
import timm
import os
from torchvision.transforms import InterpolationMode
import scipy.stats as st

img_height, img_width = 224, 224
img_max, img_min = 1., 0

cnn_model_paper = ['resnet50', 'vgg16', 'mobilenet_v2', 'inception_v3']
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224',
                   'visformer_small', 'swin_tiny_patch4_window7_224']

cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']
vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                      'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']

generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]

def load_pretrained_model(cnn_model=[], vit_model=[]):
    for model_name in cnn_model:
        yield model_name, models.__dict__[model_name](weights="DEFAULT")
        # yield model_name, models.__dict__[model_name](weights="IMAGENET1K_V1")
    for model_name in vit_model:
        yield model_name, timm.create_model(model_name, pretrained=True)


def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    model_name = model.__class__.__name__
    Resize = 224
    
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        if 'Inc' in model_name:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            Resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            Resize = 224

    PreprocessModel = PreprocessingModel(Resize, mean, std)
    return torch.nn.Sequential(PreprocessModel, model)


def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = transforms.Resize(resize)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, target_class=None, eval=False):
        self.targeted = targeted
        self.target_class = target_class
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            if self.target_class:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], self.target_class] for i in range(len(dev))}
            else:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                             dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l


# =============================
# DI/TI utility functions
# =============================

def generate_ti_kernel(kernel_type: str, kernel_size: int, device, nsig: int = 3) -> torch.Tensor:
    """
    Generate a 2D smoothing kernel for TI and stack for 3-channel conv.

    Returns: Tensor of shape [3, 1, kernel_size, kernel_size]
    """
    ktype = kernel_type.lower()
    if ktype == 'gaussian':
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
    elif ktype == 'uniform':
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel = kernel / kernel.sum()
    elif ktype == 'linear':
        kern1d = 1 - np.abs(np.linspace(-nsig, nsig, kernel_size)) / nsig
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
    else:
        raise ValueError(f'Unsupported kernel type: {kernel_type}')

    stack_kernel = np.stack([kernel, kernel, kernel])  # [3, H, W]
    stack_kernel = np.expand_dims(stack_kernel, 1)     # [3, 1, H, W]
    return torch.from_numpy(stack_kernel.astype(np.float32)).to(device)


def ti_smooth_grad(grad: torch.Tensor, kernel: torch.Tensor | None = None, enable_ti: bool = True) -> torch.Tensor:
    """
    Apply translation-invariant smoothing to gradient if enabled.
    """
    if enable_ti:
        if kernel is None:
            raise ValueError('TI is enabled but kernel is None. Ensure kernel is generated before smoothing.')
        grad = F.conv2d(grad, kernel, stride=1, padding='same', groups=3)
    return grad


def diverse_input_transform(data: torch.Tensor, di_scale_factor: float = 1.14, enable_di: bool = True) -> torch.Tensor:
    """
    Apply DI (random resize + pad) then resize back for large images.
    """
    if not enable_di:
        return data

    x_di = data
    img_width = data.size()[-1]

    enlarged_img_width = int(img_width * di_scale_factor)
    di_pad_amount = enlarged_img_width - img_width
    di_pad_value = 0
    ori_size = x_di.shape[-1]

    rnd = int(torch.rand(1) * di_pad_amount) + ori_size
    x_di = transforms.Resize((rnd, rnd), interpolation=InterpolationMode.NEAREST)(x_di)

    pad_max = ori_size + di_pad_amount - rnd
    pad_left = int(torch.rand(1) * pad_max)
    pad_right = pad_max - pad_left
    pad_top = int(torch.rand(1) * pad_max)
    pad_bottom = pad_max - pad_top

    x_di = F.pad(x_di, (pad_left, pad_right, pad_top, pad_bottom), 'constant', di_pad_value)

    if img_width > 64:
        x_di = transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(x_di)
    return x_di


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted',
                         targeted=True, eval=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break
