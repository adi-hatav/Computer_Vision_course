# Computer Vision Homework Collection

This repository contains the solutions for the Computer Vision course (EE046746) by Adi Hatav and Tomer Raz. The homework assignments explore various computer vision concepts, techniques, and algorithms. Each assignment includes a combination of classical and modern deep learning-based approaches to solving different computer vision problems.

## Homework Overview

### HW1: CIFAR10 Classification and CLIP Experiments
This assignment focuses on:
1. **Classic and CNN Classifiers**: Implemented and compared K-Nearest Neighbors (KNN) and Convolutional Neural Networks (CNN) on the CIFAR10 dataset. The CNN model achieved an accuracy of 84.77% on the test set.
2. **Foundation Models (CLIP)**: Explored zero-shot classification, clustering, and retrieval using CLIP embeddings on CIFAR10, demonstrating the generalization capabilities of CLIP.

### HW2: Keypoint Detection and BRIEF Descriptors
This assignment covered:
1. **Keypoint Detection**: Implemented keypoint detection using Difference of Gaussians (DoG).
2. **BRIEF Descriptor**: Developed a BRIEF (Binary Robust Independent Elementary Features) descriptor for matching keypoints.
3. **ORB Enhancements**: Applied ORB (Oriented FAST and Rotated BRIEF) for rotation-invariant feature extraction, improving the robustness of the matching under different rotations.

### HW3: Semantic Segmentation and Homography
This assignment included:
1. **Semantic Segmentation**: Compared classic segmentation methods (K-means, SLIC, GrabCut) and modern deep learning approaches like SAM (Segment Anything Model) on various images.
2. **Video Segmentation**: Used SAM to segment humans and dinosaurs from a video, combining them creatively.
3. **Image Stitching with Homography**: Used SIFT keypoints and RANSAC to stitch multiple images together to create panoramas, experimenting with planar homographies.

### HW4: Structure from Motion (SfM)
This assignment focused on sparse reconstruction using Structure from Motion techniques:
1. **Eight Point Algorithm**: Implemented to estimate the fundamental matrix.
2. **Epipolar Geometry**: Found corresponding points across images and computed the essential matrix.
3. **Triangulation**: Performed triangulation to reconstruct the 3D points from image pairs, and evaluated the quality using re-projection error.
4. **3D Reconstruction**: Combined these techniques to reconstruct a 3D scene from two images.

## Running the Code
Each homework folder contains Jupyter notebooks (`.ipynb`) and related files. Simply clone this repository and use Jupyter Notebook to explore the code and results.
