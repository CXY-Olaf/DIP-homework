import cv2
import numpy as np
import gradio as gr

from launch_utils import launch_demo

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None


def _tps_kernel(r2, eps=1e-8):
    return r2 * np.log(r2 + eps)


def _solve_tps(target_pts, source_pts, eps=1e-8):
    n = target_pts.shape[0]
    pairwise = target_pts[:, None, :] - target_pts[None, :, :]
    r2 = np.sum(pairwise * pairwise, axis=2)
    k_mat = _tps_kernel(r2, eps)

    p_mat = np.concatenate(
        [np.ones((n, 1), dtype=np.float64), target_pts.astype(np.float64)],
        axis=1,
    )

    system = np.zeros((n + 3, n + 3), dtype=np.float64)
    system[:n, :n] = k_mat
    system[:n, n:] = p_mat
    system[n:, :n] = p_mat.T

    rhs = np.zeros((n + 3, 2), dtype=np.float64)
    rhs[:n] = source_pts.astype(np.float64)

    try:
        params = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(system, rhs, rcond=None)[0]

    return params[:n], params[n:]


def _build_inverse_distance_map(height, width, source_pts, target_pts, alpha=1.0, eps=1e-8):
    displacement = source_pts.astype(np.float64) - target_pts.astype(np.float64)
    map_x = np.empty((height, width), dtype=np.float32)
    map_y = np.empty((height, width), dtype=np.float32)
    grid_x = np.arange(width, dtype=np.float64)

    power = max(alpha, 1e-3) / 2.0
    chunk_rows = max(1, min(height, 128))

    for start in range(0, height, chunk_rows):
        end = min(start + chunk_rows, height)
        grid_y = np.arange(start, end, dtype=np.float64)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        query = np.stack([mesh_x, mesh_y], axis=-1).reshape(-1, 2)

        diff = query[:, None, :] - target_pts[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        weights = 1.0 / np.power(dist2 + eps, power)

        exact_matches = dist2 <= eps
        if np.any(exact_matches):
            match_rows = np.where(np.any(exact_matches, axis=1))[0]
            weights[match_rows] = exact_matches[match_rows].astype(np.float64)

        weights_sum = np.sum(weights, axis=1, keepdims=True)
        mapped = query + (weights @ displacement) / np.maximum(weights_sum, eps)

        map_x[start:end] = mapped[:, 0].reshape(end - start, width).astype(np.float32)
        map_y[start:end] = mapped[:, 1].reshape(end - start, width).astype(np.float32)

    return map_x, map_y


def _build_tps_map(height, width, source_pts, target_pts, eps=1e-8):
    weights, affine = _solve_tps(target_pts, source_pts, eps=eps)
    map_x = np.empty((height, width), dtype=np.float32)
    map_y = np.empty((height, width), dtype=np.float32)
    grid_x = np.arange(width, dtype=np.float64)
    chunk_rows = max(1, min(height, 128))

    for start in range(0, height, chunk_rows):
        end = min(start + chunk_rows, height)
        grid_y = np.arange(start, end, dtype=np.float64)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        query = np.stack([mesh_x, mesh_y], axis=-1).reshape(-1, 2)

        diff = query[:, None, :] - target_pts[None, :, :]
        r2 = np.sum(diff * diff, axis=2)
        kernel = _tps_kernel(r2, eps)

        affine_term = np.concatenate(
            [np.ones((query.shape[0], 1), dtype=np.float64), query], axis=1
        )
        mapped = kernel @ weights + affine_term @ affine

        map_x[start:end] = mapped[:, 0].reshape(end - start, width).astype(np.float32)
        map_y[start:end] = mapped[:, 1].reshape(end - start, width).astype(np.float32)

    return map_x, map_y


def _status_text():
    pair_count = min(len(points_src), len(points_dst))
    pending = len(points_src) - len(points_dst)
    if image is None:
        return "请先上传图片。"
    if pair_count == 0 and pending == 0:
        return "点击左侧选点图开始标点：第 1 次点击是源点，第 2 次点击是目标点，之后交替进行。"
    if pending > 0:
        return f"已完成 {pair_count} 对点，当前等待第 {len(points_dst) + 1} 个目标点。"
    return f"已完成 {pair_count} 对点，下一次点击将记录第 {len(points_src) + 1} 个源点。"


def _draw_points(base_image):
    if base_image is None:
        return None

    marked_image = np.array(base_image).copy()
    radius = max(4, min(marked_image.shape[:2]) // 120)
    text_scale = max(0.5, radius / 8.0)
    text_thickness = max(1, radius // 3)

    for idx, pt in enumerate(points_src, start=1):
        cv2.circle(marked_image, tuple(pt), radius, (255, 0, 0), -1)
        cv2.putText(
            marked_image,
            f"S{idx}",
            (pt[0] + radius + 2, pt[1] - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )
    for idx, pt in enumerate(points_dst, start=1):
        cv2.circle(marked_image, tuple(pt), radius, (0, 0, 255), -1)
        cv2.putText(
            marked_image,
            f"T{idx}",
            (pt[0] + radius + 2, pt[1] - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    for idx in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(
            marked_image,
            tuple(points_src[idx]),
            tuple(points_dst[idx]),
            (0, 255, 0),
            max(2, radius // 2),
            tipLength=0.2,
        )

    return marked_image

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = None if img is None else np.array(img)
    return _draw_points(image), None, _status_text()

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    if image is None:
        return None, "请先上传图片，再开始选点。"

    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    return _draw_points(image), _status_text()

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """
    if image is None:
        return None

    warped_image = np.array(image)
    pair_count = min(len(source_pts), len(target_pts))
    if pair_count == 0:
        return warped_image

    source_pts = np.asarray(source_pts[:pair_count], dtype=np.float64)
    target_pts = np.asarray(target_pts[:pair_count], dtype=np.float64)

    height, width = warped_image.shape[:2]

    if pair_count == 1:
        offset = source_pts[0] - target_pts[0]
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        map_x = grid_x + np.float32(offset[0])
        map_y = grid_y + np.float32(offset[1])
    elif pair_count < 3:
        map_x, map_y = _build_inverse_distance_map(
            height, width, source_pts, target_pts, alpha=alpha, eps=eps
        )
    else:
        map_x, map_y = _build_tps_map(height, width, source_pts, target_pts, eps=eps)

    return cv2.remap(
        warped_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

def run_warping():
    global points_src, points_dst, image

    if image is None:
        return None
    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return _draw_points(image), None, _status_text()

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, type="numpy", width=800)
            point_select = gr.Image(
                label="Click to Select Source and Target Points",
                interactive=False,
                type="numpy",
                width=800,
            )
            point_status = gr.Markdown("请先上传图片。")

        with gr.Column():
            result_image = gr.Image(label="Warped Result", type="numpy", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, [point_select, result_image, point_status])
    point_select.select(record_points, None, [point_select, point_status])
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, [point_select, result_image, point_status])

if __name__ == "__main__":
    launch_demo(demo, port=7861)
