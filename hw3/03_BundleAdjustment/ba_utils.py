import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


IMAGE_SIZE = 1024
IMAGE_CENTER = IMAGE_SIZE / 2.0


def load_observations(data_dir, device="cpu"):
    data_dir = Path(data_dir)
    points2d = np.load(data_dir / "points2d.npz")
    view_names = sorted(points2d.files)
    obs = np.stack([points2d[name][:, :2] for name in view_names], axis=0).astype(np.float32)
    mask = np.stack([points2d[name][:, 2] for name in view_names], axis=0).astype(np.float32)
    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32)

    return {
        "view_names": view_names,
        "obs2d": torch.from_numpy(obs).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "colors": torch.from_numpy(colors).to(device),
        "obs2d_np": obs,
        "mask_np": mask,
        "colors_np": colors,
    }


def _constraint_row(a, b):
    return np.array(
        [
            a[0] * b[0],
            a[0] * b[1] + a[1] * b[0],
            a[0] * b[2] + a[2] * b[0],
            a[1] * b[1],
            a[1] * b[2] + a[2] * b[1],
            a[2] * b[2],
        ],
        dtype=np.float64,
    )


def _metric_upgrade(motion):
    a_rows = []
    b_vals = []
    num_views = motion.shape[0] // 2
    for i in range(num_views):
        a = motion[2 * i]
        b = motion[2 * i + 1]
        a_rows.append(_constraint_row(a, a))
        b_vals.append(1.0)
        a_rows.append(_constraint_row(b, b))
        b_vals.append(1.0)
        a_rows.append(_constraint_row(a, b))
        b_vals.append(0.0)

    a_rows = np.stack(a_rows, axis=0)
    b_vals = np.array(b_vals, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(a_rows, b_vals, rcond=None)
    l_mat = np.array(
        [
            [sol[0], sol[1], sol[2]],
            [sol[1], sol[3], sol[4]],
            [sol[2], sol[4], sol[5]],
        ],
        dtype=np.float64,
    )
    eigvals, eigvecs = np.linalg.eigh(l_mat)
    eigvals = np.clip(eigvals, 1e-6, None)
    q_mat = eigvecs @ np.diag(np.sqrt(eigvals))
    return q_mat


def factorization_initialize(obs2d, mask, focal_init=1000.0, em_iters=12, target_std=0.35):
    obs2d = np.asarray(obs2d, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    num_views, num_points, _ = obs2d.shape

    row_means = np.zeros((num_views, 2), dtype=np.float64)
    measurement = np.zeros((num_views * 2, num_points), dtype=np.float64)
    visibility = np.zeros((num_views * 2, num_points), dtype=bool)

    for i in range(num_views):
        vis = mask[i] > 0.5
        row_means[i] = obs2d[i, vis, :2].mean(axis=0)
        centered = obs2d[i, :, :2] - row_means[i]
        measurement[2 * i] = centered[:, 0]
        measurement[2 * i + 1] = centered[:, 1]
        visibility[2 * i] = vis
        visibility[2 * i + 1] = vis

    filled = measurement.copy()
    filled[~visibility] = 0.0

    for _ in range(em_iters):
        u_mat, s_vals, vh_mat = np.linalg.svd(filled, full_matrices=False)
        u3 = u_mat[:, :3]
        s3 = np.diag(np.sqrt(s_vals[:3]))
        v3 = vh_mat[:3]
        low_rank = u3 @ s3 @ s3 @ v3
        filled[~visibility] = low_rank[~visibility]
        filled[visibility] = measurement[visibility]

    u_mat, s_vals, vh_mat = np.linalg.svd(filled, full_matrices=False)
    u3 = u_mat[:, :3]
    s3 = np.diag(np.sqrt(s_vals[:3]))
    v3 = vh_mat[:3]
    motion_affine = u3 @ s3
    shape_affine = s3 @ v3

    q_mat = _metric_upgrade(motion_affine)
    q_inv = np.linalg.inv(q_mat)
    motion = motion_affine @ q_mat
    points3d = (q_inv @ shape_affine).T

    points3d -= points3d.mean(axis=0, keepdims=True)
    scale = target_std / (points3d.std() + 1e-8)
    points3d *= scale

    eulers = np.zeros((num_views, 3), dtype=np.float32)
    trans = np.zeros((num_views, 3), dtype=np.float32)
    depth_candidates = []
    rotations = []

    for i in range(num_views):
        row1 = motion[2 * i]
        row2 = motion[2 * i + 1]
        row1 = row1 / (np.linalg.norm(row1) + 1e-8)
        row2 = row2 - np.dot(row2, row1) * row1
        row2 = row2 / (np.linalg.norm(row2) + 1e-8)
        row3 = np.cross(row1, row2)
        row3 = row3 / (np.linalg.norm(row3) + 1e-8)
        r_mat = np.stack([row1, row2, row3], axis=0)
        if np.linalg.det(r_mat) < 0:
            r_mat[2] *= -1.0
        rotations.append(r_mat)

        vis = mask[i] > 0.5
        rotated = (r_mat @ points3d.T).T
        obs_centered = obs2d[i, vis] - row_means[i]
        pred_std_x = np.std(rotated[vis, 0]) + 1e-8
        pred_std_y = np.std(rotated[vis, 1]) + 1e-8
        obs_std_x = np.std(obs_centered[:, 0]) + 1e-8
        obs_std_y = np.std(obs_centered[:, 1]) + 1e-8
        scale_ratio = 0.5 * (obs_std_x / pred_std_x + obs_std_y / pred_std_y)
        depth_candidates.append(focal_init / max(scale_ratio, 1e-6))

    shared_depth = float(np.median(depth_candidates))

    for i in range(num_views):
        r_mat = rotations[i]
        eulers[i] = Rotation.from_matrix(r_mat).as_euler("xyz", degrees=False).astype(np.float32)
        mean_x, mean_y = row_means[i]
        tx = (mean_x - IMAGE_CENTER) * shared_depth / focal_init
        ty = (IMAGE_CENTER - mean_y) * shared_depth / focal_init
        trans[i] = np.array([tx, ty, -shared_depth], dtype=np.float32)

    return {
        "points3d": points3d.astype(np.float32),
        "eulers": eulers,
        "trans": trans,
        "focal": np.float32(focal_init),
        "row_means": row_means.astype(np.float32),
        "shared_depth": shared_depth,
    }


def euler_xyz_to_matrix_torch(eulers):
    cx = torch.cos(eulers[..., 0])
    cy = torch.cos(eulers[..., 1])
    cz = torch.cos(eulers[..., 2])
    sx = torch.sin(eulers[..., 0])
    sy = torch.sin(eulers[..., 1])
    sz = torch.sin(eulers[..., 2])

    one = torch.ones_like(cx)
    zero = torch.zeros_like(cx)

    rx = torch.stack(
        [
            torch.stack([one, zero, zero], dim=-1),
            torch.stack([zero, cx, -sx], dim=-1),
            torch.stack([zero, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zero, sy], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-sy, zero, cy], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zero], dim=-1),
            torch.stack([sz, cz, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )

    return rz @ ry @ rx


def save_colored_obj(points3d, colors, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points3d = np.asarray(points3d, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    with output_path.open("w", encoding="utf-8") as f_obj:
        for point, color in zip(points3d, colors):
            f_obj.write(
                f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def plot_loss_curve(loss_history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, len(loss_history) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, loss_history, linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Bundle Adjustment Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_point_cloud_preview(points3d, colors, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points3d = np.asarray(points3d, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)

    fig = plt.figure(figsize=(11, 4))
    viewpoints = [
        (20, -60, "View A"),
        (10, 0, "View B"),
        (15, 60, "View C"),
    ]
    sample = np.linspace(0, len(points3d) - 1, min(len(points3d), 8000), dtype=np.int64)

    for idx, (elev, azim, title) in enumerate(viewpoints, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.scatter(
            points3d[sample, 0],
            points3d[sample, 1],
            points3d[sample, 2],
            c=colors[sample],
            s=1.0,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(points3d.ptp(axis=0) + 1e-6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _draw_circle(draw_ctx, x, y, radius, color):
    draw_ctx.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=1)


def save_reprojection_overlays(
    image_dir,
    output_dir,
    view_names,
    obs2d,
    pred2d,
    mask,
    view_indices=(0, 12, 25, 37, 49),
    max_points=1800,
):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs2d = np.asarray(obs2d)
    pred2d = np.asarray(pred2d)
    mask = np.asarray(mask)

    for idx in view_indices:
        image = Image.open(image_dir / f"{view_names[idx]}.png").convert("RGB")
        draw = ImageDraw.Draw(image)
        visible = np.where(mask[idx] > 0.5)[0]
        if len(visible) > max_points:
            sample_idx = np.linspace(0, len(visible) - 1, max_points, dtype=np.int64)
            visible = visible[sample_idx]

        for point_idx in visible:
            ox, oy = obs2d[idx, point_idx]
            px, py = pred2d[idx, point_idx]
            _draw_circle(draw, ox, oy, radius=2, color=(0, 255, 0))
            _draw_circle(draw, px, py, radius=2, color=(255, 64, 64))

        image.save(output_dir / f"{view_names[idx]}_overlay.png")


def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_json:
        json.dump(metrics, f_json, indent=2, ensure_ascii=False)
