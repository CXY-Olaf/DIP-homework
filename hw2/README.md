# Assignment 2 - DIP with PyTorch

This repository is the submission for Homework 2 of Digital Image Processing. The homework contains a traditional image editing task based on Poisson blending and a deep learning image-to-image translation task based on a fully convolutional Pix2Pix-style model.

- Student: 程开良
- Student ID: SA25001013
- Course: 数字图像处理

## Requirements

The homework was developed and tested with the prepared conda environment:

```bash
conda activate myenv
```

Main dependencies:

- `torch`
- `numpy`
- `Pillow`
- `opencv-python`
- `gradio`

## Project Structure

```text
hw2/
├── README.md
├── verify_hw2.py
├── results/
│   ├── poisson/
│   └── pix2pix/
└── 02_DIPwithPyTorch/
    ├── README.md
    ├── reference.jpg
    ├── run_blending_gradio.py
    ├── data_poisson/
    └── Pix2Pix/
        ├── README.md
        ├── FCN_network.py
        ├── facades_dataset.py
        ├── train.py
        └── download_facades_dataset.sh
```

## Task 1: Poisson Image Editing

Implemented file:

- `02_DIPwithPyTorch/run_blending_gradio.py`

Completed items:

- polygon-to-mask conversion
- Laplacian loss computation with `torch.nn.functional.conv2d`
- Gradio interaction for polygon selection, reset, overlay preview, and blending
- UI fixes for easier point selection and faster interaction feedback

Run locally:

```bash
cd hw2/02_DIPwithPyTorch
python run_blending_gradio.py
```

Poisson blending example:

| Foreground | Background |
| --- | --- |
| ![source](results/poisson/person_source.jpg) | ![target](results/poisson/person_target.jpg) |

| Polygon Selection | Blended Result |
| --- | --- |
| ![overlay](results/poisson/overlay.png) | ![blended](results/poisson/blended.png) |

Result note:

- The selected foreground region can be placed on the target image by adjusting `dx` and `dy`.
- The current implementation focuses on completing the assignment pipeline and preserving gradient consistency inside the selected region.
- When foreground and background differ strongly in pose, illumination, and scale, the final blend may still show visible artifacts.

## Task 2: Pix2Pix-style FCN

Implemented files:

- `02_DIPwithPyTorch/Pix2Pix/FCN_network.py`
- `02_DIPwithPyTorch/Pix2Pix/facades_dataset.py`
- `02_DIPwithPyTorch/Pix2Pix/train.py`

Completed items:

- five-layer convolutional encoder
- five-layer transposed-convolution decoder
- end-to-end forward pass
- Unicode-safe dataset image loading on Windows
- configurable training arguments and checkpoint resume support

Training command used for the final run:

```bash
cd hw2/02_DIPwithPyTorch/Pix2Pix
python train.py --batch-size 16 --num-workers 0 --epochs 10 --sample-every 5 --checkpoint-every 10 --scheduler-step 10
python train.py --batch-size 16 --num-workers 0 --epochs 30 --start-epoch 10 --resume .\checkpoints\pix2pix_model_epoch_10.pth --sample-every 10 --checkpoint-every 10 --scheduler-step 30
```

Training setup:

- dataset: Facades
- batch size: 16
- num workers: 0
- optimizer: Adam
- learning rate: 0.001
- loss: L1 loss
- device: NVIDIA GeForce RTX 4060 Laptop GPU

Validation loss summary:

| Stage | Validation Loss |
| --- | ---: |
| Epoch 10 | 0.3768 |
| Best during continued training | about 0.3532 |
| Epoch 30 | 0.3586 |

Representative validation results from epoch 20:

| Result 1 | Result 2 |
| --- | --- |
| ![pix2pix result 1](results/pix2pix/facades_val_epoch20_1.png) | ![pix2pix result 2](results/pix2pix/facades_val_epoch20_2.png) |

Result note:

- The model learns the coarse semantic layout after short training.
- Outputs are still blurry and incomplete compared with the target labels.
- This is consistent with the limited training budget and the simple FCN architecture used for the homework.

## Verification

Quick smoke test:

```bash
cd hw2
python verify_hw2.py
```

This script checks:

- polygon mask generation
- Laplacian loss backward pass
- FCN forward output shape
- one-batch dataset and training smoke test

## Reproducibility Notes

- Large generated files are intentionally excluded from git with `.gitignore`, including datasets, checkpoints, training caches, and intermediate training result folders.
- The report images stored in `hw2/results/` are the curated assets intended for submission.
