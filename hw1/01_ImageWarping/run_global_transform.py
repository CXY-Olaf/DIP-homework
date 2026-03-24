import cv2
import gradio as gr
import numpy as np


def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])


def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None

    image = np.array(image)

    # Pad the image so the transformed content is less likely to be cropped.
    pad_size = min(image.shape[0], image.shape[1]) // 2
    padded = np.full(
        (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3),
        255,
        dtype=np.uint8,
    )
    padded[
        pad_size : pad_size + image.shape[0],
        pad_size : pad_size + image.shape[1],
    ] = image
    image = padded

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    m_flip = np.eye(3, dtype=np.float32)
    if flip_horizontal:
        m_flip[0, 0] = -1
        m_flip[0, 2] = w

    m_rot_scale = to_3x3(cv2.getRotationMatrix2D((cx, cy), rotation, scale)).astype(
        np.float32
    )
    m_translate = np.array(
        [[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]],
        dtype=np.float32,
    )

    # Transform order: flip -> rotate/scale -> translate.
    m_composed = m_translate @ m_rot_scale @ m_flip
    transformed_image = cv2.warpAffine(
        image,
        m_composed[:2, :],
        (w, h),
        borderValue=(255, 255, 255),
    )

    return transformed_image


def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                scale = gr.Slider(
                    minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale"
                )
                rotation = gr.Slider(
                    minimum=-180,
                    maximum=180,
                    step=1,
                    value=0,
                    label="Rotation (degrees)",
                )
                translation_x = gr.Slider(
                    minimum=-300,
                    maximum=300,
                    step=10,
                    value=0,
                    label="Translation X",
                )
                translation_y = gr.Slider(
                    minimum=-300,
                    maximum=300,
                    step=10,
                    value=0,
                    label="Translation Y",
                )
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            with gr.Column():
                image_output = gr.Image(label="Transformed Image")

        inputs = [
            image_input,
            scale,
            rotation,
            translation_x,
            translation_y,
            flip_horizontal,
        ]

        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo


if __name__ == "__main__":
    interactive_transform().launch()
