import argparse
import os
import random
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ba_utils import (
    IMAGE_CENTER,
    euler_xyz_to_matrix_torch,
    factorization_initialize,
    load_observations,
    plot_loss_curve,
    save_colored_obj,
    save_metrics,
    save_point_cloud_preview,
    save_reprojection_overlays,
)


def inverse_softplus(x):
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    return torch.where(
        x_tensor > 20.0,
        x_tensor,
        torch.log(torch.expm1(x_tensor)),
    )


class BundleAdjustmentModel(nn.Module):
    def __init__(self, init_dict, device):
        super().__init__()
        points3d = torch.from_numpy(init_dict["points3d"]).to(device)
        eulers = torch.from_numpy(init_dict["eulers"]).to(device)
        trans = torch.from_numpy(init_dict["trans"]).to(device)

        self.register_buffer("fixed_euler", eulers[:1].clone())
        self.register_buffer("fixed_trans", trans[:1].clone())

        self.euler_params = nn.Parameter(eulers[1:].clone())
        self.trans_xy = nn.Parameter(trans[1:, :2].clone())
        self.trans_z_raw = nn.Parameter(inverse_softplus((-trans[1:, 2]).clamp_min(1e-4)))
        self.points3d = nn.Parameter(points3d.clone())
        self.raw_focal = nn.Parameter(inverse_softplus(torch.tensor(float(init_dict["focal"])) ))

    def focal(self):
        return F.softplus(self.raw_focal) + 1e-4

    def camera_eulers(self):
        return torch.cat([self.fixed_euler, self.euler_params], dim=0)

    def camera_trans(self):
        trans_z = -(F.softplus(self.trans_z_raw) + 1e-4).unsqueeze(-1)
        moving = torch.cat([self.trans_xy, trans_z], dim=-1)
        return torch.cat([self.fixed_trans, moving], dim=0)

    def forward(self):
        focal = self.focal()
        eulers = self.camera_eulers()
        trans = self.camera_trans()
        rot = euler_xyz_to_matrix_torch(eulers)
        cam_points = torch.einsum("vij,nj->vni", rot, self.points3d) + trans[:, None, :]
        zc = cam_points[..., 2]
        z_safe = torch.where(zc.abs() < 1e-4, zc.sign() * 1e-4 + (zc == 0).float() * -1e-4, zc)

        u = -focal * cam_points[..., 0] / z_safe + IMAGE_CENTER
        v = focal * cam_points[..., 1] / z_safe + IMAGE_CENTER
        pred = torch.stack([u, v], dim=-1)
        return pred, zc, focal, rot, trans


def charbonnier_loss(diff, eps=1.0):
    return torch.sqrt(diff.pow(2) + eps * eps) - eps


def compute_metrics(pred2d, obs2d, mask):
    err = torch.linalg.norm(pred2d - obs2d, dim=-1)
    visible = mask > 0.5
    visible_err = err[visible]
    return {
        "mean_reprojection_error": float(visible_err.mean().item()),
        "median_reprojection_error": float(visible_err.median().item()),
        "max_reprojection_error": float(visible_err.max().item()),
        "visible_observations": int(visible.sum().item()),
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_observations(args.data_dir, device=device)
    init_dict = factorization_initialize(
        data["obs2d_np"],
        data["mask_np"],
        focal_init=args.init_focal,
        em_iters=args.init_em_iters,
        target_std=args.init_shape_std,
    )

    model = BundleAdjustmentModel(init_dict, device=device)
    obs2d = data["obs2d"]
    mask = data["mask"]

    stage1_params = [
        {"params": [model.raw_focal], "lr": args.lr_focal},
        {"params": [model.euler_params], "lr": args.lr_euler},
        {"params": [model.trans_xy, model.trans_z_raw], "lr": args.lr_trans},
    ]
    stage2_params = stage1_params + [{"params": [model.points3d], "lr": args.lr_points}]

    optimizer = torch.optim.Adam(stage1_params)
    best_state = None
    best_loss = float("inf")
    loss_history = []
    start_time = time.time()

    for step in range(1, args.num_iters + 1):
        if step == args.freeze_points_iters + 1:
            optimizer = torch.optim.Adam(stage2_params)

        optimizer.zero_grad(set_to_none=True)
        pred2d, zc, focal, rot, trans = model()
        diff = pred2d - obs2d

        reproj = (charbonnier_loss(diff, eps=args.charb_eps).sum(dim=-1) * mask).sum() / mask.sum()
        depth_penalty = (torch.relu(zc + args.depth_margin).pow(2) * mask).sum() / mask.sum()
        point_center_reg = model.points3d.mean(dim=0).pow(2).mean()
        point_scale_reg = torch.relu(model.points3d.std() - args.max_point_std).pow(2)
        smooth_rot = (model.camera_eulers()[1:] - model.camera_eulers()[:-1]).pow(2).mean()
        smooth_trans = (model.camera_trans()[1:] - model.camera_trans()[:-1]).pow(2).mean()

        loss = (
            reproj
            + args.depth_weight * depth_penalty
            + args.center_weight * point_center_reg
            + args.scale_weight * point_scale_reg
            + args.smooth_weight * (smooth_rot + smooth_trans)
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Encountered non-finite loss. Check initialization and optimization settings."
            )
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "raw_focal": model.raw_focal.detach().cpu().clone(),
                "euler_params": model.euler_params.detach().cpu().clone(),
                "trans_xy": model.trans_xy.detach().cpu().clone(),
                "trans_z_raw": model.trans_z_raw.detach().cpu().clone(),
                "points3d": model.points3d.detach().cpu().clone(),
            }

        if step % args.log_every == 0 or step == 1 or step == args.num_iters:
            print(
                f"[{step:04d}/{args.num_iters}] "
                f"loss={loss_value:.4f} "
                f"reproj={float(reproj.item()):.4f} "
                f"depth={float(depth_penalty.item()):.4f} "
                f"f={float(focal.item()):.2f}"
            )

    model.raw_focal.data.copy_(best_state["raw_focal"].to(device))
    model.euler_params.data.copy_(best_state["euler_params"].to(device))
    model.trans_xy.data.copy_(best_state["trans_xy"].to(device))
    model.trans_z_raw.data.copy_(best_state["trans_z_raw"].to(device))
    model.points3d.data.copy_(best_state["points3d"].to(device))

    with torch.no_grad():
        pred2d, zc, focal, rot, trans = model()
        metrics = compute_metrics(pred2d, obs2d, mask)
        metrics.update(
            {
                "best_training_loss": best_loss,
                "focal": float(focal.item()),
                "num_iterations": args.num_iters,
                "freeze_points_iters": args.freeze_points_iters,
                "elapsed_seconds": time.time() - start_time,
            }
        )

        points_np = model.points3d.detach().cpu().numpy()
        pred_np = pred2d.detach().cpu().numpy()
        eulers_np = model.camera_eulers().detach().cpu().numpy()
        trans_np = model.camera_trans().detach().cpu().numpy()

    plot_loss_curve(loss_history, output_dir / "loss_curve.png")
    save_point_cloud_preview(points_np, data["colors_np"], output_dir / "point_cloud_preview.png")
    save_colored_obj(points_np, data["colors_np"], output_dir / "reconstructed_points.obj")
    save_reprojection_overlays(
        Path(args.data_dir) / "images",
        output_dir / "reprojection",
        data["view_names"],
        data["obs2d_np"],
        pred_np,
        data["mask_np"],
    )
    np.savez(
        output_dir / "ba_result.npz",
        points3d=points_np,
        pred2d=pred_np,
        eulers=eulers_np,
        trans=trans_np,
        focal=np.array([metrics["focal"]], dtype=np.float32),
    )
    save_metrics(metrics, output_dir / "metrics.json")

    print("Training done.")
    print(f"Results written to: {output_dir}")
    print(f"Mean reprojection error: {metrics['mean_reprojection_error']:.4f}px")
    print(f"Median reprojection error: {metrics['median_reprojection_error']:.4f}px")


def build_argparser():
    parser = argparse.ArgumentParser(description="Bundle Adjustment for hw3")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/ba_run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iters", type=int, default=900)
    parser.add_argument("--freeze-points-iters", type=int, default=160)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--init-focal", type=float, default=1000.0)
    parser.add_argument("--init-em-iters", type=int, default=12)
    parser.add_argument("--init-shape-std", type=float, default=0.35)

    parser.add_argument("--lr-focal", type=float, default=1e-2)
    parser.add_argument("--lr-euler", type=float, default=6e-3)
    parser.add_argument("--lr-trans", type=float, default=4e-3)
    parser.add_argument("--lr-points", type=float, default=1e-3)

    parser.add_argument("--charb-eps", type=float, default=1.0)
    parser.add_argument("--depth-margin", type=float, default=0.15)
    parser.add_argument("--depth-weight", type=float, default=0.05)
    parser.add_argument("--center-weight", type=float, default=1e-3)
    parser.add_argument("--scale-weight", type=float, default=1e-3)
    parser.add_argument("--smooth-weight", type=float, default=1e-4)
    parser.add_argument("--max-point-std", type=float, default=0.9)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
