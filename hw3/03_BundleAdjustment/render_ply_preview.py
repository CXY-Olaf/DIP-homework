import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def load_binary_ply_vertices(path):
    path = Path(path)
    with path.open("rb") as f_ply:
        vertex_count = None
        format_name = None
        vertex_properties = []
        in_vertex_block = False
        while True:
            line = f_ply.readline()
            if not line:
                raise ValueError("Invalid PLY file: missing end_header")
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("format "):
                format_name = text.split()[1]
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
                in_vertex_block = True
                continue
            if text.startswith("element ") and not text.startswith("element vertex"):
                in_vertex_block = False
            if in_vertex_block and text.startswith("property "):
                parts = text.split()
                if len(parts) != 3:
                    raise ValueError(f"Unsupported PLY property declaration: {text}")
                _, dtype_name, prop_name = parts
                if dtype_name not in PLY_DTYPES:
                    raise ValueError(f"Unsupported PLY dtype: {dtype_name}")
                vertex_properties.append((prop_name, PLY_DTYPES[dtype_name]))
            if text == "end_header":
                break

        if vertex_count is None:
            raise ValueError("Invalid PLY file: missing vertex count")
        if format_name != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {format_name}")
        if not vertex_properties:
            raise ValueError("Invalid PLY file: missing vertex properties")

        dtype = np.dtype([(name, "<" + dtype_code) for name, dtype_code in vertex_properties])
        vertices = np.fromfile(f_ply, dtype=dtype, count=vertex_count)
    return vertices


def extract_xyz_rgb(vertices):
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    finite_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite_mask]

    if {"red", "green", "blue"}.issubset(vertices.dtype.names):
        rgb = np.stack(
            [vertices["red"], vertices["green"], vertices["blue"]],
            axis=1,
        ).astype(np.float32)
        rgb = rgb[finite_mask] / 255.0
    else:
        rgb = np.full((len(xyz), 3), 0.7, dtype=np.float32)

    if len(xyz) == 0:
        raise ValueError("PLY vertex array does not contain any finite 3D points")

    return xyz, rgb


def save_preview(vertices, output_path):
    xyz, rgb = extract_xyz_rgb(vertices)

    sample = np.linspace(0, len(xyz) - 1, min(len(xyz), 5000), dtype=np.int64)
    xyz = xyz[sample]
    rgb = rgb[sample]

    fig = plt.figure(figsize=(11, 4))
    views = [(18, -55, "View A"), (10, 0, "View B"), (15, 55, "View C")]
    for idx, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=2.0, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect(np.ptp(xyz, axis=0) + 1e-6)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_stats(vertices, output_path):
    xyz, _ = extract_xyz_rgb(vertices)
    stats = {
        "num_points": int(len(xyz)),
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
        "xyz_std": xyz.std(axis=0).tolist(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--stats", required=True)
    args = parser.parse_args()

    vertices = load_binary_ply_vertices(args.ply)
    save_preview(vertices, args.preview)
    save_stats(vertices, args.stats)


if __name__ == "__main__":
    main()
