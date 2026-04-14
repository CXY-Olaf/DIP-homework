from pathlib import Path
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
POISSON_DIR = ROOT / "02_DIPwithPyTorch"
PIX2PIX_DIR = POISSON_DIR / "Pix2Pix"


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_poisson():
    blending_module = load_module("run_blending_gradio", POISSON_DIR / "run_blending_gradio.py")

    points = np.array([[4, 4], [28, 4], [28, 28], [4, 28]], dtype=np.int64)
    mask = blending_module.create_mask_from_points(points, 32, 32)
    assert mask.shape == (32, 32)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0

    foreground = torch.rand(1, 3, 32, 32)
    blended = torch.rand(1, 3, 32, 32, requires_grad=True)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
    loss = blending_module.cal_laplacian_loss(foreground, mask_tensor, blended, mask_tensor)
    assert torch.is_tensor(loss)
    loss.backward()
    assert blended.grad is not None

    print(f"[PASS] Poisson mask/loss check. mask_sum={int(mask.sum())}, loss={float(loss):.6f}")


def verify_pix2pix():
    dataset_module = load_module("facades_dataset", PIX2PIX_DIR / "facades_dataset.py")
    network_module = load_module("FCN_network", PIX2PIX_DIR / "FCN_network.py")

    model = network_module.FullyConvNetwork()
    sample_input = torch.randn(1, 3, 256, 256)
    sample_output = model(sample_input)
    assert sample_output.shape == (1, 3, 256, 256)
    print(
        "[PASS] FCN forward check. "
        f"shape={tuple(sample_output.shape)}, "
        f"range=({float(sample_output.min()):.4f}, {float(sample_output.max()):.4f})"
    )

    train_list = PIX2PIX_DIR / "train_list.txt"
    if not train_list.exists():
        print("[SKIP] Dataset smoke check skipped because train_list.txt was not found.")
        return

    dataset = dataset_module.FacadesDataset(str(train_list))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch_input, batch_target = next(iter(dataloader))
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

    batch_output = model(batch_input)
    loss = criterion(batch_output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        "[PASS] Dataset/train smoke check. "
        f"input={tuple(batch_input.shape)}, target={tuple(batch_target.shape)}, loss={float(loss):.6f}"
    )


if __name__ == "__main__":
    verify_poisson()
    verify_pix2pix()
