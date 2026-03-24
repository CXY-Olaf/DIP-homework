# Assignment 1 - Image Warping

This repository is the submission for Homework 1 of Digital Image Processing. The assignment focuses on image warping and includes both global geometric transformation and point-guided image deformation.

- Student: 程开良
- Student ID: SA25001013
- Course: 数字图像处理

<img src="01_ImageWarping/pics/teaser.png" alt="teaser" width="800">

## Requirements

To install requirements:

```bash
cd 01_ImageWarping
python -m pip install -r requirements.txt
```

Main dependencies:

- `opencv-python`
- `numpy`
- `gradio`

## Running

To run the basic geometric transformation demo:

```bash
cd 01_ImageWarping
python run_global_transform.py
```

This demo supports:

- Scaling
- Rotation
- Translation
- Horizontal flip

To run the point-guided image deformation demo:

```bash
cd 01_ImageWarping
python run_point_transform.py
```

This demo supports:

- Uploading an image
- Selecting source and target control points interactively
- Visualizing point pairs and arrow directions
- Generating warped results with RBF/Thin Plate Spline based deformation

## Results

The homework contains two completed parts:

| Part | File | Status | Description |
| --- | --- | --- | --- |
| Basic geometric transformation | `run_global_transform.py` | Completed | Composition of scaling, rotation, translation, and horizontal flip around the image center |
| Point-guided deformation | `run_point_transform.py` | Completed | Point-based image warping using RBF/Thin Plate Spline mapping |

### Basic Transformation

<img src="01_ImageWarping/pics/global_demo.gif" alt="global transform demo" width="800">

### Point Guided Deformation

<img src="01_ImageWarping/pics/point_demo.gif" alt="point deformation demo" width="800">

## Project Structure

```text
hw1/
├── README.md
└── 01_ImageWarping/
    ├── README.md
    ├── requirements.txt
    ├── run_global_transform.py
    ├── run_point_transform.py
    └── pics/
        ├── teaser.png
        ├── global_demo.gif
        └── point_demo.gif
```

## Notes

- `run_global_transform.py` performs affine transform composition on a padded image to reduce cropping during transformation.
- `run_point_transform.py` implements inverse coordinate mapping and interpolation for control-point-based image warping.
- Both parts are complete and can be run independently.

## Acknowledgement

- [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf)
- [Image Warping by Radial Basis Functions](https://www.sci.utah.edu/~gerig/CS6640-F2010/Project3/Arad-1995.pdf)
- [OpenCV Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
- [Gradio Documentation](https://www.gradio.app/)
