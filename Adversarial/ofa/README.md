# OFA Attack

This repository has been reduced to the Orthogonal Feature Attack (OFA) implementation.

## Files

- `transferattack/methods/ofa.py`: OFA algorithm.
- `transferattack/attack.py`: base attack class and model loading.
- `transferattack/utils.py`: preprocessing, dataset, evaluation model, DI/TI helpers.
- `main.py`: OFA generation/evaluation entry point.

## Usage

The input directory should contain `labels.csv` and an `images/` subdirectory.

```bash
python3 main.py --model resnet50 --input_dir ./data
python3 main.py --model resnet50 --input_dir ./data --eval
```

Common OFA options:

```bash
python3 main.py --model resnet50 --K 5 --gamma 2.0 --n_components 32 --layer_names layer1,layer3
python3 main.py --model resnet50 --enable_ti --enable_di
```
