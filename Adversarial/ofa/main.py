import argparse
import os
import random
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

from transferattack.methods.ofa import OFA
from transferattack.utils import (
    AdvDataset,
    cnn_model_paper,
    load_pretrained_model,
    save_images,
    vit_model_paper,
    wrap_model,
)

ATTACK_NAME = 'ofa'


def get_parser():
    parser = argparse.ArgumentParser(description='Generate and evaluate OFA adversarial examples')
    parser.add_argument('-e', '--eval', action='store_true', help='evaluate generated adversarial examples')
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--batchsize', default=32, type=int, help='batch size')
    parser.add_argument('--model', default='resnet50', type=str, help='source surrogate model')
    parser.add_argument('--input_dir', default='./data', type=str, help='benign image dataset directory')
    parser.add_argument('--output_dir', default=None, type=str, help='adversarial image output/evaluation directory')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--gpu', default='0', type=str)

    parser.add_argument('--eps', default=16 / 255, type=float, help='perturbation budget')
    parser.add_argument('--alpha', default=1 / 255, type=float, help='update step size')
    parser.add_argument('--epoch', default=100, type=int, help='number of attack iterations')
    parser.add_argument('--momentum', default=1.0, type=float, help='momentum decay factor')
    parser.add_argument('--random_start', action='store_true', help='set random start')

    parser.add_argument('--enable_ti', action='store_true', help='enable translation-invariant gradient smoothing')
    parser.add_argument('--kernel_type', default='gaussian', type=str, help='TI kernel type: gaussian|uniform|linear')
    parser.add_argument('--kernel_size', default=5, type=int, help='TI kernel size')
    parser.add_argument('--enable_di', action='store_true', help='enable diverse input transformation')
    parser.add_argument('--di_scale_factor', default=1.14, type=float, help='DI scale factor (e.g., 340/299 ~= 1.14)')

    parser.add_argument('--K', default=5, type=int, help='OFA gradient aggregation count')
    parser.add_argument('--gamma', default=2.0, type=float, help='OFA feature decay factor')
    parser.add_argument('--disable_feature_decay', action='store_true', help='disable OFA feature decay')
    parser.add_argument('--n_components', default=32, type=int, help='OFA PCA component count')
    parser.add_argument('--layer_names', default='layer1,layer3', type=str, help='comma-separated layer names for OFA')

    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    dir_base = (
        f"{args.model}_targeted={int(args.targeted)}_"
        f"di={int(args.enable_di)}_ti={int(args.enable_ti)}"
    )
    default_output_dir = os.path.join('adv_data', ATTACK_NAME, dir_base)

    if not args.eval:
        args.output_dir = args.output_dir or default_output_dir
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    else:
        if args.output_dir is None:
            args.output_dir = default_output_dir
        if not os.path.isdir(args.output_dir):
            raise FileNotFoundError(
                f"Eval folder not found: {args.output_dir}. Please ensure it exists and contains images.")

    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    if not args.eval:
        attacker = OFA(
            model_name=args.model,
            epsilon=args.eps,
            alpha=args.alpha,
            epoch=args.epoch,
            decay=args.momentum,
            targeted=args.targeted,
            random_start=args.random_start,
            K=args.K,
            gamma=args.gamma,
            enable_feature_decay=not args.disable_feature_decay,
            n_components=args.n_components,
            layer_names=[name.strip() for name in args.layer_names.split(',') if name.strip()],
            enable_ti=args.enable_ti,
            kernel_type=args.kernel_type,
            kernel_size=args.kernel_size,
            enable_di=args.enable_di,
            di_scale_factor=args.di_scale_factor,
        )

        for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
            perturbations = attacker(images, labels)
            save_images(args.output_dir, images + perturbations.cpu(), filenames)
    else:
        res = '|'
        for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper):
            model = wrap_model(model.eval().cuda())
            for p in model.parameters():
                p.requires_grad = False
            asr = eval(model, dataloader, args.targeted)
            print(f'{model_name}: {asr:.1f}')
            res += f' {asr:.1f} |'

        print(res)
                
                
def eval(model, dataloader, is_targeted):
    correct, total = 0, 0
    for images, labels, _ in dataloader:
        if is_targeted:
            labels = labels[1]
        pred = model(images.cuda())
        correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
        total += labels.shape[0]
    if is_targeted:
        # correct: pred == target_label
        asr = (correct / total) * 100
    else:
        # correct: pred == original_label
        asr = (1 - correct / total) * 100
    return asr


if __name__ == '__main__':
    main()
